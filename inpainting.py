import sys
sys.path.append('..')
import json

import os
from datetime import datetime
from typing import Dict, List
import platform
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
import numpy as np
import fire
from tqdm import tqdm

from models.downsampler import Downsampler

from utils.common_utils import get_noise, np_to_torch
from utils.bayesian_utils import NLLLoss, uncert_regression_gal, add_noise_sgld
from train_utils import closure, track_training, get_imgs, save_run, get_net_and_optim, get_mc_preds, track_uncert_sgld


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False

num_input_channels = 1
num_channels_down = [16, 32, 64, 128, 128, 128]
num_channels_up = [16, 32, 64, 128, 128, 128]
num_channels_skip = 0
# num_scales = 5
upsample_mode = 'nearest'
filter_size_down = 5
filter_size_up = 3
filter_size_skip = 1
need1x1_up = False
# need_sigmoid = False
pad = 'reflection'

imsize = -1 # (320, 320)

mc_iter = 10
reg_noise_std = 0.
exp_weight = 0.99


def inpainting(exp_name: str = None,
               img_name: str = 'skin_lesion',
               criterion: str = 'nll',
               num_iter: int = 50000,
               num_scales: int = 6,
               gpu: int = 0,
               seed: int = 42,
               net_specs: dict = {},
               optim_specs: dict = None,
               path_log_dir: str = None,
               save: bool = True,
               net: nn.Module = None,
               optimizer: Optimizer = None) -> Dict[str, List[float]]:

    """
    Params
    ------------------------------------
    img_name:
    criterion: nll or mse
    num_scales:
    gpu:
    seed:
    net_specs: dropout_type, dropout_p, prior_mu, prior_sigma, prior_pi, kl_type, beta_type, sgld, burnin_iter, mcmc_iter
    """

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    device = 'cuda:' + str(gpu)
    # torch.set_num_threads(1)

    log_dir = [f'{str(k)}_{str(v)[:4]}' for k, v in net_specs.items()]
    if exp_name is None:
        exp_name = f"%s_%s_%s" % ('_'.join(log_dir), img_name, datetime.now().strftime("%m_%d_%Y_%H_%M_%S"))

    if path_log_dir is None:
        path_log_dir = '/media/fastdata/toelle/logs_midl_inp/%s' % exp_name
    else:
        path_log_dir = '%s/%s' % (path_log_dir, exp_name)

    if save:
        if not os.path.exists(path_log_dir):
            os.mkdir(path_log_dir)
        with open(path_log_dir + '/net_info.json', 'w') as f:
            info = net_specs.copy()
            info["num_scales"] = num_scales
            info["criterion"] = criterion
            info["img_name"] = img_name
            info["imsize"] = imsize
            json.dump(info, f, indent=4)

    imgs = get_imgs(img_name, 'inpainting', imsize=imsize)

    net_input = get_noise(num_input_channels, 'noise', (imgs['gt'].shape[1], imgs['gt'].shape[2])).to(device).detach()#.type(dtype).detach()

    num_output_channels = imgs['gt'].shape[0] + 1

    img_torch = np_to_torch(imgs['gt']).to(device)#.type(dtype)
    img_mask_torch = np_to_torch(imgs['mask']).to(device)#.type(dtype)

    if net is None and optimizer is None:
        net, optimizer = get_net_and_optim(num_input_channels, num_output_channels, num_channels_down, num_channels_up, num_channels_skip, num_scales, filter_size_down, filter_size_up, filter_size_skip, upsample_mode=upsample_mode, pad=pad, need1x1_up=need1x1_up, net_specs=net_specs, optim_specs=optim_specs)

    net = net.to(device)#.type(dtype)

    if criterion == 'nll':
        criterion = NLLLoss(reduction='mean').to(device)#.type(dtype)
    else:
        criterion = nn.MSELoss(reduction='mean').to(device)#.type(dtype)

    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

    out_avg = None
    results = {}
    sgld_imgs = [] if "sgld_cheng" in list(net_specs.keys()) else None

    pbar = tqdm(range(1, num_iter+1))
    for i in pbar:

        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        ELBO, out, _ = closure(net, optimizer, net_input, img_torch, criterion, mask=img_mask_torch)

        if out_avg is None:
            out_avg = out.detach()
        else:
            out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

        results = track_training(img_torch, img_torch*img_mask_torch, dict(to_corrupted=out, to_gt=out, to_gt_sm=out_avg), results)

        if "sgld_cheng" in list(net_specs.keys()):
            sgld_imgs = track_uncert_sgld(sgld_imgs=sgld_imgs, iter=i, img=out.detach(), **net_specs)
            add_noise_sgld(net, 2 * optim_specs["lr"])

        pbar.set_description('I: %d | ELBO: %.2f | PSNR_noisy: %.2f | PSNR_gt: %.2f | PSNR_gt_sm: %.2f' % (i, ELBO.item(), results['psnr_corrupted'][-1], results['psnr_gt'][-1], results['psnr_gt_sm'][-1]))

    if save:
        save_run(results, net, optimizer, net_input_saved, out_avg, sgld_imgs, path=path_log_dir)

    if __name__ != "__main__":
        return results


if __name__ == "__main__":
    fire.Fire(inpainting)
