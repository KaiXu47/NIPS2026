master_port=29508
export CUDA_VISIBLE_DEVICES=4
task=1-1
python -m torch.distributed.launch --nproc_per_node=1 --master_port=${master_port} scripts/dist_train_voc_seg_neg.py --step 0 --max_iters 4000 --lr 6e-5 --task ${task} --work_dir output_voc
for t in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19; do
  python -m torch.distributed.launch --nproc_per_node=1 --master_port=${master_port} scripts/dist_train_voc_seg_neg.py --step ${t} --max_iters 2000 --lr 2e-5 --task ${task} --work_dir output_voc
done
