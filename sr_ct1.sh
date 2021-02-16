#!/bin/sh

python super_resolution.py --img_name ct1 --num_iter 50000 --gpu 1 --exp_name sr_dip_ct1 --criterion mse --optim_specs '{"lr": 3e-3}' --factor 2

python super_resolution.py --img_name ct1 --gpu 1 --num_iter 50000 --factor 2 --net_specs '{"dropout_type": "2d", "dropout_p": 0.3}' --optim_specs '{"lr": 3e-3, "weight_decay": 1e-4}' --criterion nll --exp_name sr_mcd_ct1

python super_resolution.py --img_name ct1 --net_specs '{"sgld_cheng": True, "burnin_iter": 7000, "mcmc_iter": 50}' --optim_specs '{"lr": 3e-3, "weight_decay": 5e-8}' --gpu 1 --num_iter 50000 --factor 2 --criterion mse --exp_name sr_sgld_ct1


# python denoising.py --net_specs '{"prior_mu": 0., "prior_sigma": 0.05, "kl_type": "forward", "beta": 1e-6}' --optim_specs '{"lr": 0.01}' --num_iter 10 --img_name oct --exp_name mfvi_oct_100k --gpu $1 --path_log_dir $2
#
# python denoising.py --net_specs '{"dropout_type": "2d", "dropout_p": 0.3}' --optim_specs '{"lr": 3e-4, "weight_decay": 1e-4}' --seed 42 --num_iter 10 --img_name oct --exp_name mcd_oct_100k --gpu $1 --path_log_dir $2 --img_name oct
#
# python denoising.py --net_specs '{"dropout_type": "2d", "dropout_p": 0.3}' --optim_specs '{"lr": 3e-4, "weight_decay": 1e-4}' --seed 42 --num_iter 10 --img_name oct --exp_name mcd_us_100k --gpu $1 --path_log_dir $2 --img_name us
