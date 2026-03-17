import datetime
import logging
import torch.distributed as dist
from datasets import ade as ade
import os.path as osp
import tasks
from model.model_seg_neg import network
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from utils import evaluate, optimizer
from utils.pyutils import AverageMeter, cal_eta, format_tabs
import os
import torch
import torch.nn.functional as F
import numpy as np
from model.backbone.vit import replace_mlp_with_cms, remap_cms_state_dict, sync_cms_weights
from utils.modification import *

class Trainer:
    def __init__(self, args):
        self.args = args
        self.step = args.step
        self.task = args.task
        
        task_classes = tasks.get_per_task_classes(
            args.dataset, args.task, args.step
        )
        
        self.model = network(
            backbone=args.backbone,
            num_classes=sum(task_classes),
            classes_list=task_classes,
            pretrained=args.pretrained,
            init_momentum=args.momentum,
            aux_layer=args.aux_layer,
            step = args.step + 1
        )
        self.device = torch.device(args.local_rank)
        # ADE total classes are 150 (mapped to 0-149).
        # sum(task_classes) will be 100 in step 0, 110 in step 1, etc.
        self.total_classes = sum(task_classes) # Changed: total classes observed so far
        self.new_classes = task_classes[-1]
        self.old_classes = self.total_classes - self.new_classes

        if args.step == 0:
            self.model_old = None
        else:
            old_task_classes = tasks.get_per_task_classes(
                args.dataset, args.task, args.step - 1
            )
            self.model_old = network(
                backbone=args.backbone,
                num_classes=sum(old_task_classes),
                classes_list=old_task_classes,
                pretrained=args.pretrained,
                init_momentum=args.momentum,
                aux_layer=args.aux_layer,
                step = args.step
            )
            if args.step >= 2:
                self.model_old = replace_mlp_with_cms(
                    self.model_old, step=self.step
                )

            for par in self.model_old.parameters():
                par.requires_grad = False
            self.model_old.eval()

            self.model = replace_mlp_with_cms(self.model, step=self.step)

        # 加载离线相似度矩阵用于动态 Margin
        if os.path.exists("datasets/ade/clip_similarity_matrix.npy"):
            self.clip_sim_matrix = torch.from_numpy(np.load("datasets/ade/clip_similarity_matrix.npy")).cuda()
            logging.info("[!] Successfully loaded CLIP similarity matrix for Dynamic Margin.")
        else:
            self.clip_sim_matrix = None
            logging.warning("[!] CLIP similarity matrix not found. Using fixed margin.")

        param_groups = self.model.get_param_groups()
        self.optimizer = self.get_optimizer(args, param_groups)

    def get_optimizer(self, args, param_groups):
        optim_params = []
        
        from model.backbone.vit import get_cms_param_groups
        # Use LLRD (get_cms_param_groups) for all steps (including step 0)
        cms_param_groups = get_cms_param_groups(self.model.encoder, args.lr)
        for group in cms_param_groups:
            group['weight_decay'] = args.wt_decay
            optim_params.append(group)
        
        optim_params.append({
            "params": param_groups[1],
            "lr": args.lr * 10,
            "weight_decay": args.wt_decay,
        })

        optim = getattr(optimizer, args.optimizer)(
            params=optim_params,
            lr=args.lr,
            weight_decay=args.wt_decay,
            betas=args.betas,
            max_iter=args.max_iters,
            power=args.power)

        return optim

    def load_step_ckpt(self, path, step_0_checkpoints=None):
        if osp.exists(path):
            step_checkpoint = torch.load(path, map_location="cpu")
            if self.step == 1:
                state_dict = remap_cms_state_dict(step_checkpoint['model_state'], self.step)
                self.model.load_state_dict(state_dict, strict=False)
                self.model_old.load_state_dict(
                    step_checkpoint['model_state'], strict=True
                )  
            else:
                state_dict = remap_cms_state_dict(
                    step_checkpoint['model_state'], step=self.step
                )
                self.model.load_state_dict(state_dict, strict=False)
                self.model_old.load_state_dict(
                    step_checkpoint['model_state'], strict=True
                )
                self.model = sync_cms_weights(self.model, sync_mode='none')

            logging.info(f"[!] Previous model loaded from {path}")
            del step_checkpoint['model_state']
        elif osp.exists(step_0_checkpoints):
            step_checkpoint = torch.load(step_0_checkpoints, map_location="cpu")
            state_dict = remap_cms_state_dict(
                step_checkpoint['model_state'], step=self.step
            )
            self.model.load_state_dict(state_dict, strict=False)
            self.model_old.load_state_dict(
                step_checkpoint['model_state'], strict=True
            )
            logging.info(f"[!] step_0 model loaded from {step_0_checkpoints}")
            del step_checkpoint['model_state']
        else:
            logging.info(f"[!] WARNING: Unable to find step {self.args.step - 1} checkpoint!")

    def validate(self, model=None, data_loader=None, args=None):
        preds, gts = [], []
        model.eval()

        # Define TTA scales and flip
        if args.ms_val:
            scales = [0.75, 1.0, 1.25]
            flip = True
        else:
            scales = [1.0]
            flip = False

        with torch.no_grad():
            for i, data in tqdm(
                enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="
            ):
                name, inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()
                
                h, w = labels.shape[1], labels.shape[2]
                # Initialize ensemble logits on GPU
                ensemble_logits = torch.zeros((1, self.total_classes, h, w)).cuda()

                for s in scales:
                    # 1. Resize input for current scale
                    # For ViT, we align the size to 16 for better patch embedding interpolation
                    new_h = int(args.crop_size * s // 16) * 16
                    new_w = int(args.crop_size * s // 16) * 16
                    
                    input_scale = F.interpolate(
                        inputs, size=(new_h, new_w), 
                        mode='bilinear', align_corners=False
                    )

                    # 2. Inference - Original
                    _, logits, _, _, _, _ = model(input_scale)
                    logits = F.interpolate(
                        logits, size=(h, w), 
                        mode='bilinear', align_corners=False
                    )
                    ensemble_logits += F.softmax(logits, dim=1)

                    # 3. Inference - Flip
                    if flip:
                        input_flip = torch.flip(input_scale, dims=[3])
                        _, logits_flip, _, _, _, _ = model(input_flip)
                        logits_flip = torch.flip(logits_flip, dims=[3]) # Flip output back
                        logits_flip = F.interpolate(
                            logits_flip, size=(h, w), 
                            mode='bilinear', align_corners=False
                        )
                        ensemble_logits += F.softmax(logits_flip, dim=1)

                final_segs_pred = torch.argmax(ensemble_logits, dim=1)

                preds += list(final_segs_pred.cpu().numpy().astype(np.int16))
                gts += list(labels.cpu().numpy().astype(np.int16))

        # We evaluate on classes seen so far: 0 to total_classes-1
        seg_score = evaluate.scores(gts, preds, self.total_classes)
        model.train()

        tab_results = format_tabs(
            [seg_score],
            name_list=["Seg_Pred"],
            cat_list=ade.class_list
        )

        return tab_results

    def train(self, args):
        torch.cuda.set_device(args.local_rank)
        logging.info(
            "Total gpus: %d, samples per gpu: %d..."
            % (dist.get_world_size(), args.spg)
        )

        time0 = datetime.datetime.now()
        time0 = time0.replace(microsecond=0)

        train_dataset = ade.ADE20KTrainDataset(
            root_dir=args.data_folder,
            name_list_dir=args.list_folder,
            split=args.train_set,
            stage='train',
            aug=True,
            crop_size=args.crop_size,
            ignore_index=args.ignore_index,
            tasks=args.task,
            step=args.step,
            scales=args.scales,
        )

        val_dataset = ade.ADE20KSegDataset(
            root_dir=args.data_folder,
            name_list_dir=args.list_folder,
            split=args.val_set,
            stage='val',
            aug=False,
            ignore_index=args.ignore_index,
            tasks=args.task,
            step=args.step,
            scales=args.scales,
        )

        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.spg,
            num_workers=args.num_workers,
            pin_memory=False,
            drop_last=True,
            sampler=train_sampler,
            prefetch_factor=4)

        val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=args.num_workers,
                                pin_memory=False,
                                drop_last=False)

        device = self.device
        model = self.model.to(device)
        optim = self.optimizer
        logging.info('\nOptimizer: \n%s' % optim)
        model = DistributedDataParallel(
            model, device_ids=[args.local_rank], find_unused_parameters=True
        )
        train_sampler.set_epoch(np.random.randint(args.max_iters))
        train_loader_iter = iter(train_loader)
        avg_meter = AverageMeter()
        
        # Mixed precision GradScaler
        scaler = torch.cuda.amp.GradScaler()

        if self.step == 0:
            for n_iter in range(args.max_iters):
                try:
                    name, inputs, labels = next(train_loader_iter)
                except StopIteration:
                    train_sampler.set_epoch(np.random.randint(args.max_iters))
                    train_loader_iter = iter(train_loader)
                    name, inputs, labels = next(train_loader_iter)

                inputs = inputs.to(device, non_blocking=True)
                labels = labels.cuda().to(torch.long)
                
                # Filter classes not in current step
                filter_idx = labels >= self.total_classes
                labels[filter_idx] = 255
                
                with torch.cuda.amp.autocast():
                    fmap, type_seg, P, delta_p, p_final, _ = model(inputs)
                    # 主分割损失 (CE + Dice) - 模型内部已处理 Logit Scale
                    type_seg = F.interpolate(
                        type_seg, size=labels.shape[1:], mode='bilinear', 
                        align_corners=False
                    )
                    proto_seg_loss = get_type_seg_loss(
                        type_seg, labels, 
                        ignore_index=args.ignore_index
                    )     
                    dice_loss = get_dice_loss(
                        type_seg, labels, 
                        ignore_index=args.ignore_index
                    )
                    if self.clip_sim_matrix is not None:
                        # 使用动态 Margin：由 CLIP 相似度决定，上限设为 0.6
                        proto_sep_loss = compute_dynamic_margin_ortho_loss(
                            P, self.clip_sim_matrix[:P.size(0), :P.size(0)], max_margin=0.6
                        )
                    else:
                        proto_sep_loss = compute_comprehensive_ortho_loss(
                            P, p_final, delta_p, mode='all', margin=0.2
                        )
                    
                    # 组合损失：主损失 (1.0) + 分离损失 (0.1)
                    loss = 0.1 * proto_sep_loss + \
                           1.0 * (proto_seg_loss + dice_loss)

                optim.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()

                avg_meter.add({
                    'proto_seg_loss': proto_seg_loss,
                    'dice_loss': dice_loss,
                    'proto_sep_loss': proto_sep_loss,
                })

                if (n_iter + 1) % args.log_iters == 0:
                    delta, eta = cal_eta(time0, n_iter + 1, args.max_iters)
                    cur_lr = optim.param_groups[0]['lr']
                    if args.local_rank == 0:
                        logging.info(
                            "Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; "
                            "seg: %.4f, dice: %.4f, sep: %.4f" % (
                                n_iter + 1, delta, eta, cur_lr,
                                avg_meter.pop('proto_seg_loss'),
                                avg_meter.pop('dice_loss'),
                                avg_meter.pop('proto_sep_loss'),
                            )
                        )

                if (n_iter + 1) % args.eval_iters == 0:
                    ckpt_name = os.path.join(
                        args.ckpt_dir, "model_iter_%d.pth" % (n_iter + 1)
                    )
                    if args.local_rank == 0:
                        logging.info('Validating...')
                        if args.save_ckpt:
                            torch.save(model.state_dict(), ckpt_name)
                    tab_results = self.validate(
                        model=model, data_loader=val_loader, args=args
                    )
                    if args.local_rank == 0:
                        logging.info("\n" + tab_results)

        else:
            model_old = self.model_old.to(device)            
            for n_iter in range(args.max_iters):
                try:
                    img_name, inputs, label = next(train_loader_iter)
                except StopIteration:
                    train_sampler.set_epoch(np.random.randint(args.max_iters))
                    train_loader_iter = iter(train_loader)
                    img_name, inputs, label = next(train_loader_iter)

                inputs = inputs.to(device, non_blocking=True)
                label = label.to(device, dtype=torch.long, non_blocking=True)
                
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        old_fmap, type_seg_old, P_old, \
                        delta_p_old, p_final_old, _ = model_old(inputs)

                    fmap, type_seg_new, P_new, delta_p_new, p_final_new, _ = model(inputs)

                    old_segs = F.interpolate(
                        type_seg_old, size=label.shape[1:], mode='bilinear', align_corners=False
                    )
                    old_pixel_label = torch.argmax(old_segs, dim=1)

                    filter_old_idx = label < self.old_classes
                    filter_idx = label >= self.total_classes
                    # ADE has no background class 0, so we ignore old classes in hard loss
                    # and rely on distillation/pseudo-labeling if implemented.
                    label[filter_old_idx] = 255 
                    label[filter_idx] = 255

                    mixed_pseudo_label = label.clone()
                    # ADE20K: label 255 is ignore/void, classes are 0-149.
                    # old_pixel_label from old model will predict 0-(old_classes-1).
                    mask_old = (label == 255) & (old_pixel_label < self.old_classes)
                    mixed_pseudo_label[mask_old] = old_pixel_label[mask_old]

                    type_seg_new = F.interpolate(
                        type_seg_new, size=mixed_pseudo_label.shape[1:],
                        mode='bilinear', align_corners=False
                    )
                    proto_seg_loss = get_type_seg_loss(
                        type_seg_new, mixed_pseudo_label,
                        ignore_index=args.ignore_index
                    )
                    dice_loss = get_dice_loss(
                        type_seg_new, mixed_pseudo_label,
                        ignore_index=args.ignore_index
                    )
                    
                    proto_kd_loss = prototype_kd_loss(P_new, P_old, skip_bg=False)
                    if self.clip_sim_matrix is not None:
                        proto_sep_loss = compute_dynamic_margin_ortho_loss(
                            P_new, self.clip_sim_matrix[:P_new.size(0), :P_new.size(0)], max_margin=0.6
                        )
                    else:
                        proto_sep_loss = compute_comprehensive_ortho_loss(
                            P_new, p_final_new, delta_p_new, mode='all', margin=0.2
                        )

                    prototype_loss = 1 * (proto_seg_loss + dice_loss) + \
                                     0.1 * proto_kd_loss + 0.1 * proto_sep_loss

                    loss = prototype_loss

                if (n_iter + 1) % args.eval_iters == 0:
                    if args.local_rank == 0:
                        state = {"model_state": self.model.state_dict()}
                        path = os.path.join(args.ckpt_dir, f"model_{n_iter+1}.pth")
                        torch.save(state, path)
                        logging.info(f"[!] Checkpoint saved: {path}")

                avg_meter.add({
                    'proto_seg_loss': proto_seg_loss.item(),
                    'dice_loss': dice_loss.item(),
                    'proto_kd_loss': proto_kd_loss.item(),
                    'proto_sep_loss': proto_sep_loss.item(),
                })
                optim.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()

                if (n_iter + 1) % args.log_iters == 0:
                    delta, eta = cal_eta(time0, n_iter + 1, args.max_iters)
                    cur_lr = optim.param_groups[0]['lr']
                    if args.local_rank == 0:
                        logging.info(
                            "Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; "
                            "seg: %.4f, dice: %.4f, kd: %.4f, sep: %.4f" % (
                                n_iter + 1, delta, eta, cur_lr,
                                avg_meter.pop('proto_seg_loss'),
                                avg_meter.pop('dice_loss'),
                                avg_meter.pop('proto_kd_loss'),
                                avg_meter.pop('proto_sep_loss')
                            )
                        )

                if (n_iter + 1) % args.eval_iters == 0:
                    ckpt_name = os.path.join(args.ckpt_dir, 
                    "model_iter_%d.pth" % (n_iter + 1))
                    if args.local_rank == 0:
                        logging.info('Validating...')
                        if args.save_ckpt:
                            torch.save(model.state_dict(), ckpt_name)
                    tab_results = self.validate(model=model, 
                        data_loader=val_loader, args=args)
                    if args.local_rank == 0:
                        logging.info("\n" + tab_results)

        return True
