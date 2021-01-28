#!/bin/sh

gpu=0
num_iter_eval_fn=10000
n_random=5
trials=15
save_trials=True
config=bo

for metric in "uce" "lpips" "psnr_gt" "psnr_corrupted"
do
    python bo.py --metric $metric --task $1 --num_iter_eval_fn $num_iter_eval_fn --n_random $n_random --trials $trials --config $config --save_trials $save_trials --gpu $gpu
done
