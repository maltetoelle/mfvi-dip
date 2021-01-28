#!/bin/sh

gpu=0
num_iter=10000

python inpainting.py --img_name skin_lesion --num_iter $num_iter --gpu $gpu --net_specs '{"prior_mu": 0.0, "prior_sigma": 0.1, "kl_type": "forward", "beta": 1e-6}' --optim_specs '{"lr": 0.01}'

python inpainting.py --img_name skin_lesion2 --num_iter $num_iter --gpu $gpu --optim_specs '{"lr": 0.01}'

python inpainting.py --img_name skin_lesion2 --num_iter $num_iter --gpu $gpu --net_specs '{"prior_mu": 0.0, "prior_sigma": 0.1, "kl_type": "forward", "beta": 1e-6}' --optim_specs '{"lr": 0.01}'
