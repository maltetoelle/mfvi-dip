#!/bin/sh

python inpainting.py --img_name skin_lesion1 --net_specs '{"sgld_cheng": True, "burnin_iter": 7000, "mcmc_iter": 50}' --optim_specs '{"lr": 3e-3, "weight_decay": 5e-8}' --exp_name inp_sgld_skin_lesion1 --num_iter 50000 --gpu 1

python inpainting.py --img_name skin_lesion1 --net_specs '{"dropout_type": "2d", "dropout_p": 0.3}' --optim_specs '{"lr": 3e-3, "weight_decay": 1e-4}' --exp_name inp_mcd_skin_lesion1_1 --num_iter 50000 --gpu 1

python inpainting.py --img_name skin_lesion1 --net_specs '{}' --optim_specs '{"lr": 3e-3}' --num_iter 50000 --gpu 1 --exp_name inp_dip_skin_lesion1_1
