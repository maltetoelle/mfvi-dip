#!/bin/sh

img_name=skin_lesion2
num_iter=50000

python3 inpainting.py --exp_name inp_dip_skin --img_name $img_name --net_specs '{}' --optim-specs '{"lr": 5e-3}' --num_iter $num_iter --criterion mse --path_log_dir ${1-experiments} --gpu ${2-0}

python3 inpainting.py --exp_name inp_sgld_skin --img_name $img_name --net_specs '{"sgld_cheng": True, "burnin_iter": 7000, "mcmc_iter": 50}' --optim-specs '{"lr": 5e-3, "weight_decay": 5e-8}' --num_iter $num_iter --criterion mse --path_log_dir ${1-experiments} --gpu ${2-0}

python3 inpainting.py --exp_name inp_mcd_skin --img_name $img_name --net_specs '{"dropout_type": "2d", "dropout_p": 0.3}' --optim-specs '{"lr": 5e-3, "weight_decay": 1e-4}' --num_iter $num_iter --criterion nll --path_log_dir ${1-experiments} --gpu ${2-0}

python3 inpainting.py --exp_name inp_mfvi_skin --img_name $img_name --net_specs '{"prior_mu": 0.0, "prior_sigma": 0.36, "kl_type": "forward", "beta": 1e-6}' --optim-specs '{"lr": 5e-3}' --num_iter $num_iter --criterion mse --path_log_dir ${1-experiments} --gpu ${2-0}
