#!/bin/sh

num_iter_eval_fn=10000
n_random=5
trials=15
save_trials=True
config=bo

for metric in "uce" "lpips" "psnr_gt"
do
    python bo.py --metric $metric --task denoising --num_iter_eval_fn $num_iter_eval_fn --n_random $n_random --trials $trials --config $config --save_trials $save_trials --gpu 1
done
