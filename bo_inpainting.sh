#!/bin/sh

trials=10
num_iter_eval_fn=5
num_iter_gp=1000
metric=psnr_gt_sm
batch_size=3

python3 bo_dip.py --exp_name bo_inp --task inpainting --img_name skin_lesion --trials $trials --num_iter_eval_fn $num_iter_eval_fn --num_iter_gp $num_iter_gp --metric $metric --batch_size $batch_size --log_dir ${1-./experiments} --gpus ${2-'[0]'}
