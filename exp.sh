#!/bin/sh

num_iter=1000

net_specs_mfvi='{"prior_mu": 0.,"prior_sigma": 0.1,"kl_type": "forward","beta": 1e-6}'
optim_specs_mfvi='{"lr": 0.01}'

net_specs_sgld='{"sgld": True, "burnin_iter": 100, "mcmc_iter": 10}'
optim_specs_sgld='{"lr": 0.01, "weight_decay": 5e-8}'

net_specs_dip='{}'
optim_specs_dip='{"lr": 0.01}'

net_specs_mc='{"dropout_type": "2d", "dropout_p": 0.3}'
optim_specs_mc='{"lr": 0.01, "weight_decay": 3e-4}'


python denoising.py --net_specs $net_specs_mfvi --optim_specs $optim_specs_mfvi --num_iter num_iter

python super_resolution.py --net_specs $net_specs_mfvi --optim_specs $optim_specs_mfvi --num_iter num_iter

python inpainting.py --net_specs $net_specs_mfvi --optim_specs $optim_specs_mfvi --num_iter num_iter


python denoising.py --net_specs $net_specs_sgld --optim_specs $optim_specs_sgld --num_iter num_iter

python super_resolution.py --net_specs $net_specs_sgld --optim_specs $optim_specs_sgld --num_iter num_iter

python inpainting.py --net_specs $net_specs_sgld --optim_specs $optim_specs_sgld --num_iter num_iter


python denoising.py --net_specs $net_specs_dip --optim_specs $optim_specs_dip --num_iter num_iter

python super_resolution.py --net_specs $net_specs_dip --optim_specs $optim_specs_dip --num_iter num_iter

python inpainting.py --net_specs $net_specs_dip --optim_specs $optim_specs_dip --num_iter num_iter


python denoising.py --net_specs $net_specs_mc --optim_specs $optim_specs_mc --num_iter num_iter

python super_resolution.py --net_specs $net_specs_mc --optim_specs $optim_specs_mc --num_iter num_iter

python inpainting.py --net_specs $net_specs_mc --optim_specs $optim_specs_mc --num_iter num_iter
