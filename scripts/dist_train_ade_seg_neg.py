import argparse
import datetime
import logging
import os
import random
import sys
import warnings
import numpy as np
import torch
from torch import distributed

sys.path.insert(0, ".")
import tasks
from utils.pyutils import setup_logger
from continual.Trainer_ade import Trainer

warnings.filterwarnings("ignore")
torch.hub.set_dir("./pretrained")
parser = argparse.ArgumentParser()

parser.add_argument(
    "--backbone", default="vit_base_patch16_384", type=str, help="vit_base_patch16_384"
)
parser.add_argument(
    "--pooling", default="gmp", type=str, help="pooling choice for patch tokens"
)
parser.add_argument(
    "--pretrained",
    default=True,
    type=bool,
    help="use imagenet pretrained weights",
)

parser.add_argument(
    "--task",
    type=str,
    default="100-10",
    choices=tasks.get_task_list(),
    help="Task to be executed (default: 100-10)",
)
parser.add_argument("--step", default=0, type=int, help="training_step")

parser.add_argument("--dataset", type=str, default='ade', help='Name of dataset')
parser.add_argument(
    "--data_folder", default="/data/DatasetCollection/ADEChallengeData2016", type=str, help="dataset folder"
)
parser.add_argument(
    "--list_folder", default="datasets/ade", type=str, help="train/val/test list file"
)
parser.add_argument("--num_classes", default=150, type=int, help="number of classes")
parser.add_argument("--crop_size", default=512, type=int, help="crop_size in training")
parser.add_argument(
    "--local_crop_size", default=96, type=int, help="crop_size for local view"
)
parser.add_argument("--ignore_index", default=255, type=int, help="random index")

parser.add_argument(
    "--work_dir", default="output_ade", type=str, help="work_dir_ade"
)
parser.add_argument(
    "--cfg_name", default="high_value_filter", type=str, help="cfg_name"
)

parser.add_argument("--train_set", default="train", type=str, help="training split")
parser.add_argument("--val_set", default="val", type=str, help="validation split")
parser.add_argument("--spg", default=4, type=int, help="samples_per_gpu")
parser.add_argument("--scales", default=(0.5, 2), help="random rescale in training")

parser.add_argument(
    "--optimizer", default="PolyWarmupAdamW", type=str, help="optimizer"
)
parser.add_argument("--lr", default=2e-5, type=float, help="learning rate")
parser.add_argument("--wt_decay", default=1e-2, type=float, help="weights decay")
parser.add_argument("--betas", default=(0.9, 0.95), help="betas for Adam")
parser.add_argument(
    "--power", default=0.9, type=float, help="poweer factor for poly scheduler"
)

parser.add_argument("--max_iters", default=20000, type=int, help="max training iters")
parser.add_argument("--log_iters", default=50, type=int, help=" logging iters")
parser.add_argument("--eval_iters", default=4000, type=int, help="validation iters")

parser.add_argument("--w_seg", default=0.1, type=float, help="w_seg")
parser.add_argument("--w_kd", default=0.2, type=float, help="w_kd")

parser.add_argument("--temp", default=0.5, type=float, help="temp")
parser.add_argument("--momentum", default=0.9, type=float, help="temp")
parser.add_argument("--aux_layer", default=-3, type=int, help="aux_layer")

parser.add_argument("--seed", default=0, type=int, help="fix random seed")
parser.add_argument("--save_ckpt", action="store_true", help="save_ckpt")
parser.add_argument("--ms_val", action="store_true", help="enable multi-scale validation")

parser.add_argument("--local-rank", default=0, type=int, help="local_rank")
parser.add_argument("--num_workers", default=8, type=int, help="num_workers")
parser.add_argument('--backend', default='nccl')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def save_ckpt(path, trainer):
    state = {
        "model_state": trainer.model.state_dict(),
    }
    path = os.path.join(path, "model_final.pth")
    torch.save(state, path)

if __name__ == "__main__":
    distributed.init_process_group(backend='nccl', init_method='env://')
    args = parser.parse_args()
    timestamp = "{0:%Y-%m-%d-%H-%M-%S-%f}".format(datetime.datetime.now())
    
    args.work_dir = os.path.join(args.work_dir, args.task)
    task_path = args.work_dir
    args.work_dir = os.path.join(args.work_dir, args.cfg_name)
    
    load_ckpt_dir = os.path.join(
        args.work_dir, "step" + str(args.step - 1), "checkpoints"
    )
    args.work_dir = os.path.join(args.work_dir, "step"+str(args.step))
    args.ckpt_dir = os.path.join(args.work_dir, "checkpoints")
    args.pred_dir = os.path.join(args.work_dir, "predictions")

    if args.local_rank == 0:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.makedirs(args.pred_dir, exist_ok=True)
        setup_logger(filename=os.path.join(args.work_dir, f'train_{timestamp}.log'))
        logging.info('Pytorch version: %s' % torch.__version__)
        logging.info("GPU type: %s"%(torch.cuda.get_device_name(0)))
        logging.info('\nargs: %s' % args)

    setup_seed(args.seed)
    trainer = Trainer(args=args)
    
    if args.step > 0:
        trainer.load_step_ckpt(
            os.path.join(load_ckpt_dir, "model_final.pth"),
            os.path.join(task_path, "model_final.pth"),
        )
        
    Flag = trainer.train(args=args)
    
    if args.local_rank == 0 and Flag:
        save_ckpt(args.ckpt_dir, trainer)
        logging.info("[!] Checkpoint saved.")
