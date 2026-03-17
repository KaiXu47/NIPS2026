master_port=29509
export CUDA_VISIBLE_DEVICES=0
task=offline
python_path=/home/fangkai/miniconda3/envs/py39/bin/python

# Full training (150 classes)
$python_path -m torch.distributed.launch --nproc_per_node=1 --master_port=${master_port} scripts/dist_train_ade_seg_neg.py --step 0 --max_iters 80000 --lr 6e-5 --task ${task} --work_dir output_ade_offline --spg 8
