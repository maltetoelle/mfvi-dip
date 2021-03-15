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
from utils.bayesian_utils import NLLLoss, add_noise_sgld, uncert_regression_gal
from train_utils import closure, track_training, get_imgs, save_run, get_net_and_optim, get_mc_preds, track_uncert_sgld


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False

num_input_channels = 32
num_channels_down = 128
num_channels_up = 128
num_channels_skip = 4
num_scales = 5
upsample_mode = 'bilinear'

imsize = -1 # (320, 320)
enforce_div32 = 'CROP'

mc_iter = 10
exp_weight = 0.99

def super_resolution(exp_name: str = None,
                     img_name: str = 'mri',
                     criterion: str = 'nll',
                     num_iter: int = 50000,
                     factor: int = 2,
                     kernel_type: str = 'lanczos2',
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
    # bay_inf_type:
    criterion: nll or mse
    # lr:
    # weight_decay:
    factor:
    kernel_type:
    gpu:
    seed:
    net_specs: dropout_type, dropout_p, prior_mu, prior_sigma, prior_pi, kl_type, beta_type
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
        path_log_dir = '/media/fastdata/toelle/logs_midl_sr/%s' % exp_name
    else:
        path_log_dir = '%s/%s' % (path_log_dir, exp_name)

    if save:
        if not os.path.exists(path_log_dir):
            os.mkdir(path_log_dir)
        with open(path_log_dir + '/net_info.json', 'w') as f:
            info = net_specs.copy()
            info["criterion"] = criterion
            info["factor"] = factor
            info["kernel"] = kernel_type
            info["imsize"] = imsize
            info["img_name"] = img_name
            json.dump(info, f, indent=4)

    imgs = get_imgs(img_name, 'super_resolution', imsize=imsize, factor=factor, enforce_div32=enforce_div32)

    # if factor == 4:
    reg_noise_std = 0.03
    # elif factor == 8:
    #     reg_noise_std = 0.05
    # else:
    #     assert False, 'We did not experiment with other factors'
    
    net_input = get_noise(num_input_channels, 'noise', (imgs['HR_np'].shape[1], imgs['HR_np'].shape[2]))#.type(dtype).detach()

    num_output_channels = imgs['HR_np'].shape[0] + 1

    net_input = net_input.to(device).detach()
    img_LR_var = np_to_torch(imgs['LR_np']).to(device)#.type(dtype)
    img_HR_var = np_to_torch(imgs['HR_np']).to(device)#.type(dtype)

    downsampler = Downsampler(n_planes=imgs['LR_np'].shape[0]+1, factor=factor, kernel_type=kernel_type, kernel_width=2*factor, sigma=0.5, phase=0.5, preserve_size=True).to(device)#.type(dtype)

    if net is None and optimizer is None:
        net, optimizer = get_net_and_optim(num_input_channels, num_output_channels, num_channels_down, num_channels_up, num_channels_skip, num_scales, net_specs=net_specs, optim_specs=optim_specs)

    # net = net.type(dtype)
    net = net.to(device)

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

        ELBO, out_HR, out_LR = closure(net, optimizer, net_input, img_LR_var, criterion, downsampler)

        if out_avg is None:
            out_avg = out_HR.detach()
        else:
            out_avg = out_avg * exp_weight + out_HR.detach() * (1 - exp_weight)

        results = track_training(img_LR_var, img_HR_var, dict(to_corrupted=out_LR, to_gt=out_HR, to_gt_sm=out_avg), results)

        if "sgld_cheng" in list(net_specs.keys()):
            sgld_imgs = track_uncert_sgld(sgld_imgs=sgld_imgs, iter=i, img=out_LR.detach(), **net_specs)
            add_noise_sgld(net, 2 * optim_specs["lr"])

        pbar.set_description('I: %d | ELBO: %.2f | PSNR_LR: %.2f | PSNR_HR: %.2f | PSNR_HR_gt_sm: %.2f' % (i, ELBO.item(), results['psnr_corrupted'][-1], results['psnr_gt'][-1], results['psnr_gt_sm'][-1]))

    if save:
        save_run(results, net, optimizer, net_input_saved, out_avg, sgld_imgs, path=path_log_dir)

    if __name__ != "__main__":
        return results


if __name__ == '__main__':
    fire.Fire(super_resolution)
