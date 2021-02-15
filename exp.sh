#!/bin/sh

# python inpainting.py --net_specs '{"dropout_type": "2d", "dropout_p": 0.3}' --optim_specs '{"lr": 3e-4, "weight_decay": 1e-4}' --exp_name inp_mcd_skin_lesion0 --gpu 0 --criterion nll --img_name skin_lesion0

python inpainting.py --net_specs '{"dropout_type": "2d", "dropout_p": 0.3}' --optim_specs '{"lr": 3e-4, "weight_decay": 1e-4}' --exp_name inp_mcd_skin_lesion1 --gpu 0 --criterion nll --img_name skin_lesion1 --num_iter 10000

python inpainting.py --net_specs '{"dropout_type": "2d", "dropout_p": 0.3}' --optim_specs '{"lr": 3e-4, "weight_decay": 1e-4}' --exp_name inp_mcd_skin_lesion2 --gpu 0 --criterion nll --img_name skin_lesion2 --num_iter 10000

python inpainting.py --net_specs '{"dropout_type": "2d", "dropout_p": 0.3}' --optim_specs '{"lr": 3e-4, "weight_decay": 1e-4}' --exp_name inp_mcd_skin_lesion3 --gpu 0 --criterion nll --img_name skin_lesion3 --num_iter 10000

python inpainting.py --net_specs '{"dropout_type": "2d", "dropout_p": 0.3}' --optim_specs '{"lr": 3e-4, "weight_decay": 1e-4}' --exp_name inp_mcd_skin_lesion4 --gpu 0 --criterion nll --img_name skin_lesion4 --num_iter 10000

python inpainting.py --net_specs '{"dropout_type": "2d", "dropout_p": 0.3}' --optim_specs '{"lr": 3e-4, "weight_decay": 1e-4}' --exp_name inp_mcd_skin_lesion5 --gpu 0 --criterion nll --img_name skin_lesion5 --num_iter 10000

# python denoising.py --net_specs '{"sgld_cheng": True, "burnin_iter": 7000, "mcmc_iter": 50}' --optim_specs '{"lr": 3e-4, "weight_decay": 1e-4}' --num_iter 100000 --exp_name den_sgld_xray_100k --gpu 1 --criterion mse
#
# python super_resolution.py --net_specs '{"sgld_cheng": True, "burnin_iter": 7000, "mcmc_iter": 50}' --optim_specs '{"lr": 3e-4, "weight_decay": 1e-4}' --exp_name sr_sgld_mr0 --gpu 1 --criterion mse

# python inpainting.py --net_specs '{"sgld_cheng": True, "burnin_iter": 7000, "mcmc_iter": 50}' --optim_specs '{"lr": 3e-4, "weight_decay": 1e-4}' --exp_name inp_sgld_skin_lesion0 --gpu 1 --criterion mse


# python denoising.py --net_specs '{}' --optim_specs '{"lr": 3e-4}' --num_iter 100000 --exp_name den_dip_xray_100k --gpu 1 --num_iter 100000 --criterion mse
#
# python super_resolution.py --net_specs '{}' --optim_specs '{"lr": 3e-4}' --exp_name sr_dip_mr0 --gpu 1 --criterion mse

python inpainting.py --net_specs '{}' --optim_specs '{"lr": 3e-4}' --exp_name inp_dip_skin_lesion1 --gpu 0 --criterion mse --img_name skin_lesion1 --num_iter 20000

python inpainting.py --net_specs '{}' --optim_specs '{"lr": 3e-4}' --exp_name inp_dip_skin_lesion2 --gpu 0 --criterion mse --img_name skin_lesion2 --num_iter 20000

python inpainting.py --net_specs '{}' --optim_specs '{"lr": 3e-4}' --exp_name inp_dip_skin_lesion3 --gpu 0 --criterion mse --img_name skin_lesion3 --num_iter 20000

python inpainting.py --net_specs '{}' --optim_specs '{"lr": 3e-4}' --exp_name inp_dip_skin_lesion4 --gpu 0 --criterion mse --img_name skin_lesion4 --num_iter 20000

python inpainting.py --net_specs '{}' --optim_specs '{"lr": 3e-4}' --exp_name inp_dip_skin_lesion5 --gpu 0 --criterion mse --img_name skin_lesion5 --num_iter 20000


python inpainting.py --net_specs '{"sgld_cheng": True, "burnin_iter": 7000, "mcmc_iter": 50}' --optim_specs '{"lr": 3e-4, "weight_decay": 1e-4}' --exp_name inp_mcd_skin_lesion1 --gpu 0 --criterion mse --img_name skin_lesion1 --num_iter 15000

python inpainting.py --net_specs '{"sgld_cheng": True, "burnin_iter": 7000, "mcmc_iter": 50}' --optim_specs '{"lr": 3e-4, "weight_decay": 1e-4}' --exp_name inp_mcd_skin_lesion2 --gpu 0 --criterion mse --img_name skin_lesion2 --num_iter 15000

python inpainting.py --net_specs '{"sgld_cheng": True, "burnin_iter": 7000, "mcmc_iter": 50}' --optim_specs '{"lr": 3e-4, "weight_decay": 1e-4}' --exp_name inp_mcd_skin_lesion3 --gpu 0 --criterion mse --img_name skin_lesion3 --num_iter 15000

python inpainting.py --net_specs '{"sgld_cheng": True, "burnin_iter": 7000, "mcmc_iter": 50}' --optim_specs '{"lr": 3e-4, "weight_decay": 1e-4}' --exp_name inp_mcd_skin_lesion4 --gpu 0 --criterion mse --img_name skin_lesion4 --num_iter 15000

python inpainting.py --net_specs '{"sgld_cheng": True, "burnin_iter": 7000, "mcmc_iter": 50}' --optim_specs '{"lr": 3e-4, "weight_decay": 1e-4}' --exp_name inp_mcd_skin_lesion5 --gpu 0 --criterion mse --img_name skin_lesion5 --num_iter 15000


# python denoising.py --net_specs $net_specs_mc --optim_specs $optim_specs_mc --num_iter num_iter --num_iter 100000

# python super_resolution.py --net_specs '{"dropout_type": "2d", "dropout_p": 0.3}' --optim_specs '{"lr": 3e-4, "weight_decay": 1e-4}' --exp_name sr_mcd_mr0 --gpu 1
#
# python inpainting.py --net_specs '{"dropout_type": "2d", "dropout_p": 0.3}' --optim_specs '{"lr": 3e-4, "weight_decay": 1e-4}' --exp_name inp_mcd_skin_lesion0 --gpu 1
