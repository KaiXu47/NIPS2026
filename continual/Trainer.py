import datetime
import logging
import os
import numpy as np
import torch.distributed as dist
from datasets import voc as voc
import os.path as osp
import tasks
from model.losses import get_masked_ptc_loss, get_seg_loss
from model.model_seg_neg import network
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from model.PAR import PAR
from utils import evaluate, imutils, optimizer
from utils.camutils import *
from utils.pyutils import AverageMeter, cal_eta, format_tabs
from utils.modification import *


class Trainer:
    def __init__(self, args):
        self.args = args
        self.step = args.step
        self.task = args.task
        self.model = network(
            backbone=args.backbone,
            num_classes=sum(tasks.get_per_task_classes(args.dataset, args.task, args.step)),
            classes_list=tasks.get_per_task_classes(args.dataset, args.task, args.step),
            pretrained=args.pretrained,
            init_momentum=args.momentum,
            aux_layer=args.aux_layer,
            step = args.step + 1
        )
        self.device = torch.device(args.local_rank)
        self.total_classes = sum(tasks.get_per_task_classes(args.dataset, args.task, args.step)) - 1
        self.new_classes = tasks.get_per_task_classes(args.dataset, args.task, args.step)[-1]
        self.old_classes = self.total_classes - self.new_classes
        self.new_classes_origin_weight = 0
        self.new_classes_origin_idx = 0

        if args.step == 0:  # if step 0, we don't need to instance the model_old
            self.model_old = None
        else:  # instance model_old
            self.model_old = network(
                backbone=args.backbone,
                num_classes=sum(tasks.get_per_task_classes(args.dataset, args.task, args.step - 1)),
                classes_list=tasks.get_per_task_classes(args.dataset, args.task, args.step - 1),
                pretrained=args.pretrained,
                init_momentum=args.momentum,
                aux_layer=args.aux_layer,
                step = args.step
            )
            for par in self.model_old.parameters():
                par.requires_grad = False
            self.model_old.eval()

        param_groups = self.model.get_param_groups()
        self.optimizer = self.get_optimizer(args, param_groups)

    def get_optimizer(self, args, param_groups):

        optim = getattr(optimizer, args.optimizer)(
            params=[
                {
                    "params": param_groups[0],
                    "lr": args.lr,
                    "weight_decay": args.wt_decay,
                },
                {
                    "params": param_groups[1],
                    "lr": args.lr,
                    "weight_decay": args.wt_decay,
                },
                {
                    "params": param_groups[2],
                    "lr": args.lr * 10,
                    "weight_decay": args.wt_decay,
                },
                {
                    "params": param_groups[3],
                    "lr": args.lr * 10,
                    "weight_decay": args.wt_decay,
                },
            ],
            lr=args.lr,
            weight_decay=args.wt_decay,
            betas=args.betas,
            warmup_iter=args.warmup_iters,
            max_iter=args.max_iters,
            warmup_ratio=args.warmup_lr,
            power=args.power)

        return optim

    def load_step_ckpt(self, path, step_0_checkpoints=None):
        # generate model from path
        if osp.exists(path):
            step_checkpoint = torch.load(path, map_location="cpu")
            self.model.load_state_dict(step_checkpoint['model_state'], strict=False)  # False for incr. classifiers
            self.model_old.load_state_dict(step_checkpoint['model_state'], strict=True)  # Load also here old parameters

            logging.info(f"[!] Previous model loaded from {path}")
            # clean memory
            del step_checkpoint['model_state']
        elif osp.exists(step_0_checkpoints):
            step_checkpoint = torch.load(step_0_checkpoints, map_location="cpu")
            self.model.load_state_dict(step_checkpoint['model_state'], strict=False)  # False for incr. classifiers
            self.model_old.load_state_dict(step_checkpoint['model_state'], strict=True)  # Load also here old parameters

            logging.info(f"[!] step_0 model loaded from {step_0_checkpoints}")
            del step_checkpoint['model_state']

        else:
            logging.info(f"[!] WARNING: Unable to find of step {self.args.step - 1}! "
                         f"Do you really want to do from scratch?")

    def validate(self, model=None, data_loader=None, args=None):
        preds, gts, cams, cams_aux, type_preds = [], [], [], [], []
        model.eval()
        avg_meter = AverageMeter()
        with torch.no_grad():
            for _, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):
                name, inputs, labels, cls_label, depth = data
                inputs = inputs[:,:3,:,:]
                inputs = inputs.cuda()
                labels = labels.cuda()
                cls_label = cls_label.cuda()
                cls_label = cls_label[:, :self.total_classes]

                inputs = F.interpolate(inputs, size=[args.crop_size, args.crop_size], mode='bilinear',
                                       align_corners=False)
                depth = F.interpolate(depth, size=[args.crop_size, args.crop_size], mode='bilinear',
                                       align_corners=False)

                cls, segs, _, _, type_seg,_,_,_ = model(inputs, depth)

                cls_pred = (cls > 0).type(torch.int16)
                _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
                avg_meter.add({"cls_score": _f1})

                _cams, _cams_aux = multi_scale_cam2(model, inputs, depth, args.cam_scales)
                resized_cam = F.interpolate(_cams, size=labels.shape[1:], mode='bilinear', align_corners=False)
                cam_label = cam_to_label(resized_cam, cls_label, bkg_thre=args.bkg_thre, high_thre=args.high_thre,
                                         low_thre=args.low_thre, ignore_index=args.ignore_index)

                resized_cam_aux = F.interpolate(_cams_aux, size=labels.shape[1:], mode='bilinear', align_corners=False)
                cam_label_aux = cam_to_label(resized_cam_aux, cls_label, bkg_thre=args.bkg_thre,
                                             high_thre=args.high_thre, low_thre=args.low_thre,
                                             ignore_index=args.ignore_index)

                cls_pred = (cls > 0).type(torch.int16)
                _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
                avg_meter.add({"cls_score": _f1})

                resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
                type_segs = F.interpolate(type_seg, size=labels.shape[1:], mode='bilinear', align_corners=False)
                preds += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))
                type_preds += list(torch.argmax(type_segs, dim=1).cpu().numpy().astype(np.int16))
                cams += list(cam_label.cpu().numpy().astype(np.int16))
                gts += list(labels.cpu().numpy().astype(np.int16))
                cams_aux += list(cam_label_aux.cpu().numpy().astype(np.int16))


        cls_score = avg_meter.pop('cls_score')
        seg_score = evaluate.scores(gts, preds, self.total_classes + 1)
        cam_score = evaluate.scores(gts, cams, self.total_classes + 1)
        cam_aux_score = evaluate.scores(gts, cams_aux, self.total_classes + 1)
        type_seg_score = evaluate.scores(gts, type_preds, self.total_classes + 1)
        model.train()

        tab_results = format_tabs([cam_score, cam_aux_score, seg_score, type_seg_score], name_list=["CAM", "aux_CAM", "Seg_Pred", "Type_Pred"],
                                  cat_list=voc.class_list)

        return cls_score, tab_results

    def train(self, args):
        torch.cuda.set_device(args.local_rank)
        logging.info("Total gpus: %d, samples per gpu: %d..." % (dist.get_world_size(), args.spg))

        time0 = datetime.datetime.now()
        time0 = time0.replace(microsecond=0)
        train_dataset = voc.VOC12ClsDataset(
            root_dir=args.data_folder,
            name_list_dir=args.list_folder,
            split=args.train_set,
            stage='train',
            aug=True,
            # resize_range=cfg.dataset.resize_range,
            rescale_range=args.scales,
            crop_size=args.crop_size,
            img_fliplr=True,
            ignore_index=args.ignore_index,
            num_classes=args.num_classes,
            tasks=args.task,
            step=args.step,
        )

        train_step0_dataset = voc.VOC12Step0Dataset(
            root_dir=args.data_folder,
            name_list_dir=args.list_folder,
            split=args.train_set,
            stage='train',
            aug=True,
            crop_size=args.crop_size,
            ignore_index=args.ignore_index,
            num_classes=args.num_classes,
            tasks=args.task,
            step=args.step,
        )

        val_dataset = voc.VOC12SegDataset(
            root_dir=args.data_folder,
            name_list_dir=args.list_folder,
            split=args.val_set,
            stage='val',
            aug=False,
            ignore_index=args.ignore_index,
            num_classes=args.num_classes,
            tasks=args.task,
            step=args.step,
        )

        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.spg,
            # shuffle=True,
            num_workers=args.num_workers,
            pin_memory=False,
            drop_last=True,
            sampler=train_sampler,
            prefetch_factor=4)

        train_step0_loader = DataLoader(
            train_step0_dataset,
            batch_size=args.spg,
            # shuffle=True,
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
        model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
        train_sampler.set_epoch(np.random.randint(args.max_iters))
        train_loader_iter = iter(train_loader)
        avg_meter = AverageMeter()

        train_step0_loader_iter = iter(train_step0_loader)

        if self.step == 0:

            for n_iter in range(args.max_iters):

                try:
                    name, inputs, labels, cls_label, depth = next(train_step0_loader_iter)
                except:
                    train_sampler.set_epoch(np.random.randint(args.max_iters))
                    train_step0_loader_iter = iter(train_step0_loader)
                    name, inputs, labels, cls_label, depth = next(train_step0_loader_iter)

                inputs = inputs.to(device, non_blocking=True)
                depth = depth.to(device, non_blocking=True)
                depth = depth.cuda()
                inputs = inputs.cuda()
                labels = labels.cuda()
                cls_label = cls_label.cuda()
                cls_label = cls_label[:, :self.total_classes]

                filter_idx = labels >= self.total_classes + 1
                labels[filter_idx] = 255

                #remove cow baseline
                # cls_label , labels = remove_cow(cls_label, labels)


                cls, segs, fmap, cls_aux, type_seg, P, delta_p, p_final = model(
                    inputs, depth,  # NEW
                )
                # 分割分支保持不变
                segs = F.interpolate(segs, size=[448, 448], mode='bilinear', align_corners=False)
                cls_loss = F.multilabel_soft_margin_loss(cls, cls_label)
                cls_loss_aux = F.multilabel_soft_margin_loss(cls_aux, cls_label)
                seg_loss = get_seg_loss(segs, labels.type(torch.long), ignore_index=args.ignore_index)
                

                T = 0.1
                type_seg = F.interpolate(type_seg, size=labels.shape[1:], mode='bilinear', align_corners=False)
                proto_seg_loss = get_type_seg_loss(type_seg / T, labels.type(torch.long), ignore_index=args.ignore_index)
                
                proto_sep_loss = compute_comprehensive_ortho_loss(P, p_final, delta_p, mode= 'final_only')




                loss = 1 * cls_loss + 1 * cls_loss_aux + args.w_seg * seg_loss + 0.1 * proto_sep_loss + 0.2 * proto_seg_loss

                optim.zero_grad()
                loss.backward()
                optim.step()

                # 记录
                avg_meter.add({
                    'cls_loss': cls_loss,
                    'seg_loss': seg_loss,
                    'proto_seg_loss': proto_seg_loss,
                    'proto_sep_loss': proto_sep_loss,
                })

                if (n_iter + 1) % args.log_iters == 0:
                    delta, eta = cal_eta(time0, n_iter + 1, args.max_iters)
                    cur_lr = optim.param_groups[0]['lr']
                    # print(model.module.prototype_module())
                    if args.local_rank == 0:
                        logging.info(
                            "Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; cls_loss: %.4f ,seg_loss: %.4f, proto_seg_loss: %.4f..., proto_sep_loss: %.4f..." % (
                                n_iter + 1, delta, eta, cur_lr,
                                avg_meter.pop('cls_loss'), avg_meter.pop('seg_loss'),avg_meter.pop('proto_seg_loss'),avg_meter.pop('proto_sep_loss'),
                            ))

                if (n_iter + 1) % args.eval_iters == 0:
                    ckpt_name = os.path.join(args.ckpt_dir, "model_iter_%d.pth" % (n_iter + 1))
                    if args.local_rank == 0:
                        logging.info('Validating...')
                        if args.save_ckpt:
                            torch.save(model.state_dict(), ckpt_name)
                    val_cls_score, tab_results = self.validate(model=model, data_loader=val_loader, args=args)
                    if args.local_rank == 0:
                        logging.info("val cls score: %.6f" % (val_cls_score))
                        logging.info("\n" + tab_results)

        else:
            model_old = self.model_old.to(device)            

            # state_dict = torch.load("/home/zhonggai/python-work-space/WILSS/ICME/output_voc/10-5/high_value_filter/step2/checkpoints/model_final.pth", map_location="cuda")
            # self.model.load_state_dict(state_dict['model_state'], strict=True)

            par = PAR(num_iter=10, dilations=[1, 2, 4, 8, 12, 24]).cuda()

            for n_iter in range(args.max_iters):
                try:
                    img_name, inputs, cls_label, img_box, crops, depth = next(train_loader_iter)
                except:
                    train_sampler.set_epoch(np.random.randint(args.max_iters))
                    train_loader_iter = iter(train_loader)
                    img_name, inputs, cls_label, img_box, crops, depth = next(train_loader_iter)

                inputs = inputs.to(device, non_blocking=True)
                depth = depth.to(device, non_blocking=True)
                depth = depth.cuda()
                inputs_denorm = imutils.denormalize_img2(inputs.clone())

                cls_label = cls_label.to(device, non_blocking=True)
                cls_label = cls_label[:, :self.total_classes]
                old_cls, old_segs, _x4, old_cls_aux, type_seg_old, P_old, delta_p_old, p_final_old = model_old(inputs, depth=depth)
                cls_label_old_pred = (old_cls > 2.0).long()

                cls_label_gt_new = cls_label[:, -self.new_classes:]
                cls_label = torch.cat((cls_label_old_pred, cls_label_gt_new), dim=1)

                cams, cams_aux = multi_scale_cam2(model, inputs=inputs, depth=depth, scales=args.cam_scales)

                cls, segs, fmap, cls_aux, type_seg_new, P_new, delta_p_new, p_final_new = model(inputs, depth=depth)

                old_segs = F.interpolate(type_seg_old, size=[448, 448], mode='bilinear', align_corners=False)
                old_pixel_label = torch.argmax(old_segs, dim=1)
                # old_pixel_label_filter = filter_old_dense_labels(old_pixel_label, cls_label = cls_label, ignore_index=args.ignore_index)


                cls_loss = F.multilabel_soft_margin_loss(cls, cls_label)
                cls_loss_aux = F.multilabel_soft_margin_loss(cls_aux, cls_label)



                valid_cam, _ = cam_to_label(cams.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True,
                                            bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre,
                                            ignore_index=args.ignore_index)

            

                refined_pseudo_label = refine_cams_with_bkg_v2(par, inputs_denorm, cams=valid_cam, cls_labels=cls_label,
                                                               high_thre=args.high_thre, low_thre=args.low_thre,
                                                               ignore_index=0, img_box=img_box, )

            

                mixed_pseudo_label = get_mixed_label(refined_pseudo_label, old_pixel_label.clone(),
                                                     self.total_classes, self.new_classes)

                

                segs = F.interpolate(segs, size=refined_pseudo_label.shape[1:], mode='bilinear', align_corners=False)
                seg_loss = get_seg_loss(segs, mixed_pseudo_label.type(torch.long), ignore_index=args.ignore_index)

                resized_cams_aux = F.interpolate(cams_aux, size=fmap.shape[2:], mode="bilinear", align_corners=False)
                _, pseudo_label_aux = cam_to_label(resized_cams_aux.detach(), cls_label=cls_label, img_box=img_box,
                                                   ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=args.high_thre,
                                                   low_thre=args.low_thre, ignore_index=args.ignore_index)
                aff_mask = label_to_aff_mask(pseudo_label_aux)
                ptc_loss = get_masked_ptc_loss(fmap, aff_mask)



                T = 0.1
                type_seg_new = F.interpolate(type_seg_new, size=mixed_pseudo_label.shape[1:], mode='bilinear', align_corners=False)
                proto_seg_loss = get_type_seg_loss(type_seg_new / T, mixed_pseudo_label.type(torch.long), ignore_index=args.ignore_index)
                # print(proto_seg_loss)
                
                proto_kd_loss = prototype_kd_loss(P_new, P_old, cls_label, self.new_classes)
                # proto_sep_loss = prototype_sep_loss(prototype_new, cls_label, self.new_classes)

                proto_sep_loss = compute_comprehensive_ortho_loss(P_new, p_final_new, delta_p_new, mode= 'final_only')

                # prototype_peak_loss = margin_triplet_peaky_loss(type_seg_new)
                # print(prototype_peak_loss)
                prototype_loss = 2 * proto_seg_loss + 1 * proto_kd_loss + 1 * proto_sep_loss

            

                # warmup
                if n_iter <= 2000:
                    loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + args.w_ptc * ptc_loss + 0.0 * seg_loss 
                else:
                    loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + args.w_ptc * ptc_loss + args.w_seg * seg_loss + 0.1* prototype_loss

                if n_iter % 2000 == 0 and n_iter != 0:
                    if args.local_rank == 0:  # save model at the eval iteration
                        state = {
                            "model_state": self.model.state_dict(),
                        }
                        path = os.path.join(args.ckpt_dir, f"model_{n_iter}.pth")
                        torch.save(state, path)
                        logging.info("[!] Checkpoint saved.")

                cls_pred = (cls > 0).type(torch.int16)
                cls_score = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
                avg_meter.add({
                    'cls_loss': cls_loss,
                    'ptc_loss': ptc_loss,
                    'cls_loss_aux': cls_loss_aux,
                    'seg_loss': seg_loss,
                    'cls_score': cls_score,
                    'proto_seg_loss':proto_seg_loss.item(),
                    'proto_kd_loss': proto_kd_loss.item(),
                    'proto_sep_loss': proto_sep_loss.item(),
                })
                optim.zero_grad()
                loss.backward()
                optim.step()
                # model.module.prototype_module.update_prototype_with_mask(p_iter, momentum=0.98)
                if (n_iter + 1) % args.log_iters == 0:

                    delta, eta = cal_eta(time0, n_iter + 1, args.max_iters)
                    cur_lr = optim.param_groups[0]['lr']

                    if args.local_rank == 0:
                        logging.info(
                            "Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; cls_loss: %.4f, cls_loss_aux: %.4f, ptc_loss: %.4f, seg_loss: %.4f, proto_seg_loss: %.4f..., proto_kd_loss: %.4f, proto_sep_loss: %.4f..." % (
                                n_iter + 1, delta, eta, cur_lr, avg_meter.pop('cls_loss'),
                                avg_meter.pop('cls_loss_aux'),
                                avg_meter.pop('ptc_loss'), avg_meter.pop('seg_loss'),avg_meter.pop('proto_seg_loss'),avg_meter.pop('proto_kd_loss'),avg_meter.pop('proto_sep_loss')))

                if (n_iter + 1) % args.eval_iters == 0:
                    ckpt_name = os.path.join(args.ckpt_dir, "model_iter_%d.pth" % (n_iter + 1))
                    if args.local_rank == 0:
                        logging.info('Validating...')
                        if args.save_ckpt:
                            torch.save(model.state_dict(), ckpt_name)
                    val_cls_score, tab_results = self.validate(model=model, data_loader=val_loader, args=args)
                    if args.local_rank == 0:
                        logging.info("val cls score: %.6f" % (val_cls_score))
                        logging.info("\n" + tab_results)

        return True