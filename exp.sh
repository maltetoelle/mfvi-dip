#!/bin/sh

# SR MFVI

python super_resolution.py --net_specs '{"prior_mu": 0.0, "prior_sigma": 0.075, "kl_type": "forward", "beta": 1e-6}' --optim_specs '{"lr": 0.01}' --img_name mri0 --gpu $1 --path_log_dir $2 --num_iter 50000 --exp_name sr_mfvi_mri0

python super_resolution.py --net_specs '{"prior_mu": 0.0, "prior_sigma": 0.075, "kl_type": "forward", "beta": 1e-6}' --optim_specs '{"lr": 0.01}' --img_name mri1 --gpu $1 --path_log_dir $2 --num_iter 10000 --exp_name sr_mfvi_mri1

# SR MRI1

python super_resolution.py --net_specs '{"dropout_type": "2d", "dropout_p": 0.3}' --optim_specs '{"lr": 3e-3, "weight_decay": 1e-4}' --img_name mri0 --gpu $1 --path_log_dir $2 --num_iter 50000 --exp_name sr_mcd_mri0

# python super_resolution.py --net_specs '{"sgld_cheng": True, "burnin_iter": 7000, "mcmc_iter": 50}' --optim_specs '{"lr": 0.01, "weight_decay": 5e-8}' --gpu $1 --path_log_dir $2 --num_iter 50000 --img_name mri0

python super_resolution.py --net_specs '{"sgld_cheng": True, "burnin_iter": 7000, "mcmc_iter": 50}' --optim_specs '{"lr": 0.01, "weight_decay": 5e-8}' --gpu $1 --path_log_dir $2 --num_iter 50000 --img_name mri1 --criterion mse --exp_name sr_sgld_mri1

python super_resolution.py --net_specs '{"dropout_type": "2d", "dropout_p": 0.3}' --optim_specs '{"lr": 3e-3, "weight_decay": 1e-4}' --img_name mri1 --gpu $1 --path_log_dir $2 --num_iter 50000 --exp_name sr_mcd_mri1

python super_resolution.py --net_specs '{}' --optim_specs '{"lr": 0.01}' --criterion mse --gpu $1 --path_log_dir $2 --num_iter 50000 --img_name mri1 --exp_name sr_dip_mri1

# SR CT0

python super_resolution.py --net_specs '{"sgld_cheng": True, "burnin_iter": 7000, "mcmc_iter": 50}' --optim_specs '{"lr": 0.01, "weight_decay": 5e-8}' --gpu $1 --path_log_dir $2 --num_iter 50000 --img_name ct0 --criterion mse --exp_name sr_sgld_ct0

python super_resolution.py --net_specs '{"dropout_type": "2d", "dropout_p": 0.3}' --optim_specs '{"lr": 3e-3, "weight_decay": 1e-4}' --img_name ct0 --gpu $1 --path_log_dir $2 --num_iter 50000 --exp_name sr_mcd_ct0

python super_resolution.py --net_specs '{}' --optim_specs '{"lr": 0.01}' --criterion mse --gpu $1 --path_log_dir $2 --num_iter 50000 --img_name ct0 --exp_name sr_dip_ct0

# SR CT1

python super_resolution.py --net_specs '{"sgld_cheng": True, "burnin_iter": 7000, "mcmc_iter": 50}' --optim_specs '{"lr": 0.01, "weight_decay": 5e-8}' --gpu $1 --path_log_dir $2 --num_iter 50000 --img_name ct1 --criterion mse --exp_name sr_sgld_ct1

python super_resolution.py --net_specs '{"dropout_type": "2d", "dropout_p": 0.3}' --optim_specs '{"lr": 3e-3, "weight_decay": 1e-4}' --img_name ct1 --gpu $1 --path_log_dir $2 --num_iter 50000 --exp_name sr_mcd_ct1

python super_resolution.py --net_specs '{}' --optim_specs '{"lr": 0.01}' --criterion mse --gpu $1 --path_log_dir $2 --num_iter 50000 --img_name ct1 --exp_name sr_dip_ct1
