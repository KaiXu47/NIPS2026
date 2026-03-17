master_port=29502
export CUDA_VISIBLE_DEVICES=1
task=100-10
python_path=/home/fangkai/miniconda3/envs/py39/bin/python

# Step 0: Base training (100 classes)
$python_path -m torch.distributed.launch --nproc_per_node=1 --master_port=${master_port} scripts/dist_train_ade_seg_neg.py --ms_val --step 0 --max_iters 40000 --lr 1e-4 --task ${task} --work_dir output_ade --spg 8 --crop_size 512

# Incremental steps (1-5)
for t in 4 5; do
  $python_path -m torch.distributed.launch --nproc_per_node=1 --master_port=${master_port} scripts/dist_train_ade_seg_neg.py --ms_val --step ${t} --max_iters 8000 --lr 1e-5 --task ${task} --work_dir output_ade --spg 6 --crop_size 512
done
