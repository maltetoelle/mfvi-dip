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
import lpips

from models.downsampler import Downsampler

from common_utils import get_noise, np_to_torch
from train_utils import closure, track_training, get_imgs, save_run, get_net_and_optim, get_mc_preds, track_uncert_sgld

from BayTorch.inference.losses import NLLLoss2d, uceloss
from BayTorch.inference.utils import uncert_regression_gal
from BayTorch.optimizer.sgld import SGLD

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True

num_input_channels = 32
num_channels_down = 128
num_channels_up = 128
num_channels_skip = 4
num_scales = 5
upsample_mode = 'bilinear'

imsize = -1 # (320, 320)

mc_iter = 10
reg_noise_std = 1./10. # 1./30.
exp_weight = 0.99


def denoising(img_name: str = 'xray',
              criterion: str = 'nll',
              num_iter: int = 50000,
              sigma: float = 0.1,
              gpu: int = 0,
              seed: int = 42,
              net_specs: dict = {},
              optim_specs: dict = None,
              path_log_dir: str = None,
              save: bool = True,
              net: nn.Module = None,
              optimizer: Optimizer = None,
              lpips_loss: lpips.LPIPS = None) -> Dict[str, List[float]]:

    """
    Params
    ------------------------------------
    img_name:
    criterion: nll or mse
    sigma:
    gpu:
    seed:
    net_specs: dropout_type, dropout_p, prior_mu, prior_sigma, prior_pi, kl_type, beta_type, sgld, burnin_iter, mcmc_iter
    """

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = 'cuda:' + str(gpu)

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    log_dir = [f'{str(k)}_{str(v)[:4]}' for k, v in net_specs.items()]
    if path_log_dir is None:
        path_log_dir = '/media/fastdata/toelle/logs_midl_den/%s_%s_%s' % ('_'.join(log_dir), img_name, datetime.now().strftime("%m_%d_%Y_%H_%M_%S"))
    else:
        path_log_dir = '%s/%s_%s_%s' % (path_log_dir, '_'.join(log_dir), img_name, datetime.now().strftime("%m_%d_%Y_%H_%M_%S"))

    if save:
        os.mkdir(path_log_dir)
        with open(path_log_dir + '/net_info.json', 'w') as f:
            info = net_specs.copy()
            info["criterion"] = criterion
            info["sigma"] = sigma
            info["img_name"] = img_name
            info["imsize"] = imsize
            json.dump(info, f, indent=4)

    imgs = get_imgs(img_name, 'denoising', imsize=imsize, sigma=sigma)

    # net_input = get_noise(num_input_channels, 'noise', (imgs['gt'].shape[1], imgs['gt'].shape[2]))#.type(dtype).detach()
    net_input = torch.zeros((1, num_input_channels, imgs['gt'].shape[1], imgs['gt'].shape[2]))
    net_input.uniform_()
    net_input *= 0.1
    # net_input = torch.randn((1, 32, imgs['gt'].shape[1], imgs['gt'].shape[2]))

    num_output_channels = imgs['gt'].shape[0] + 1

    # img_noisy_torch = np_to_torch(imgs['noisy']).type(dtype)
    # img_torch = np_to_torch(imgs['gt']).type(dtype)

    if net is None and optimizer is None:
        net, optimizer = get_net_and_optim(num_input_channels, num_output_channels, num_channels_down, num_channels_up, num_channels_skip, num_scales, net_specs=net_specs, optim_specs=optim_specs)

        # net = net.type(dtype)
        net = net.to(device)

    net_input = net_input.to(device).detach()
    img_noisy_torch = np_to_torch(imgs['noisy']).to(device)
    img_torch = np_to_torch(imgs['gt']).to(device)

    if criterion == 'nll':
        criterion = NLLLoss2d(reduction='mean').to(device)#.type(dtype)
    else:
        criterion = nn.MSELoss(reduction='mean').to(device)#.type(dtype)

    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

    out_avg = None
    results = {}
    sgld_imgs = [] if isinstance(optimizer, SGLD) else None

    pbar = tqdm(range(1, num_iter+1))
    for i in pbar:
    # for i in range(1, num_iter+1):

        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        ELBO, out, _ = closure(net, optimizer, net_input, img_noisy_torch, criterion)

        if out_avg is None:
            out_avg = out.detach()
        else:
            out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

        results = track_training(img_noisy_torch, img_torch, dict(to_corrupted=out, to_gt=out, to_gt_sm=out_avg), results)

        if isinstance(optimizer, SGLD):
            sgld_imgs = track_uncert_sgld(sgld_imgs=sgld_imgs, iter=i, img=out.detach(), **net_specs)

        pbar.set_description('I: %d | ELBO: %.2f | PSNR_noisy: %.2f | PSNR_gt: %.2f | PSNR_gt_sm: %.2f' % (i, ELBO.item(), results['psnr_corrupted'][-1], results['psnr_gt'][-1], results['psnr_gt_sm'][-1]))
        # print('I: %d/%d | ELBO: %.2f | PSNR_noisy: %.2f | PSNR_gt: %.2f | PSNR_gt_sm: %.2f' % (i, num_iter, ELBO.item(), results['psnr_corrupted'][-1], results['psnr_gt'][-1], results['psnr_gt_sm'][-1]))
        # sys.stdout.flush()


    img_list = get_mc_preds(net, net_input, mc_iter)
    _, _, uncert = uncert_regression_gal(img_list, reduction=None)

    out_torch_mean = torch.mean(torch.cat(img_list, dim=0)[:], dim=0, keepdim=True)
    mse_err = F.mse_loss(out_torch_mean[:,:-1], img_noisy_torch, reduction='none')

    uce, err_in_bin, avg_sigma_in_bin, freq_in_bin = uceloss(mse_err, uncert, n_bins=10, outlier=0.02)
    discr_mse_uncert = torch.abs(mse_err.mean() - uncert.mean()).item()

    if lpips_loss is None:
        lpips_loss = lpips.LPIPS(net='alex').to(device)

    lpips_metric = lpips_loss(img_noisy_torch, out_torch_mean[:,:-1]).item()

    # TODO: this is just a very quick hack
    results["uce"] = [uce.item()] * 50
    results["discr_mse_uncert"] = [discr_mse_uncert] * 50
    results["lpips"] = [lpips_metric] * 50

    if save:
        save_run(results, net, optimizer, net_input_saved, out_avg, sgld_imgs, path=path_log_dir)

    if __name__ != "__main__":
        return results


if __name__ == '__main__':
    fire.Fire(denoising)
