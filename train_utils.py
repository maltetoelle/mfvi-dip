import sys
sys.path.append('..')

from typing import Union, List, Callable, Tuple, Dict

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from utils.common_utils import torch_to_np, np_to_pil, pil_to_np, get_fname, get_image, crop_image
from utils.denoising_utils import get_noisy_image
from utils.sr_utils import load_LR_HR_imgs_sr
from utils.viz_utils import np_plot, analyse_calibration, analyse_calibration_sgld

from models import skip

from BayTorch import MeanFieldVI, MCDropoutVI
from BayTorch.inference.losses import NLLLoss2d
from BayTorch.optimizer.sgld import SGLD


def closure(net: nn.Module,
            optimizer: torch.optim.Optimizer,
            inputs: Tensor,
            target: Tensor,
            criterion: nn.Module,
            post_processor: nn.Module = None,
            mask: Tensor = torch.ones(1)):

    mask = mask.to(inputs.device)
    optimizer.zero_grad()

    out = net(inputs)
    out[:,:-1] = torch.sigmoid(out[:,:-1])

    if post_processor is not None:
        out_p = post_processor(out)
    else:
        out_p = out

    if isinstance(criterion, NLLLoss2d):
        nll = criterion(out_p * mask, target * mask)
    else:
        nll = criterion(out_p[:,:-1] * mask, target * mask)

    if isinstance(net, MeanFieldVI):
        ELBO = nll + net.kl # * net.beta
    else:
        ELBO = nll

    ELBO.backward()
    optimizer.step()

    return ELBO, out, out_p


def get_mc_preds(net: nn.Module, inputs: Tensor, mc_iter: int = 25) -> List[Tensor]:

    img_list = []
    with torch.no_grad():
        for _ in range(mc_iter):
            out = net(inputs)
            out[:,:-1] = torch.sigmoid(out[:,:-1])
            out[:,-1:] = torch.exp(-torch.clamp(out[:,-1:], min=-20, max=20))
            img_list.append(out)
    return img_list


def track_training(corrupted_img: Tensor,
                   gt_img: Tensor,
                   recons: Dict[str, Tensor],
                   results: Dict[str, List] = dict()):

    with torch.no_grad():
        mse_corrupted = F.mse_loss(recons['to_corrupted'][:,:-1], corrupted_img).item()
        mse_gt = F.mse_loss(recons['to_gt'][:,:-1], gt_img).item()

    swap_channels = lambda img: np.moveaxis(img, 0, -1)
    convert_to_np = lambda img: torch_to_np(img)[0] if img.shape[0] == 1 else torch_to_np(img)

    multichannel = True if gt_img.shape[0] == 3 else False

    corrupted_img = convert_to_np(corrupted_img)
    gt_img = convert_to_np(gt_img)
    recons = {k: convert_to_np(v) for k, v in recons.items()}

    for recon in recons:
        psnr_corrupted = peak_signal_noise_ratio(corrupted_img, recons['to_corrupted'])
        psnr_gt = peak_signal_noise_ratio(gt_img, recons['to_gt'])
        psnr_gt_sm = peak_signal_noise_ratio(gt_img, recons['to_gt_sm'])

        if multichannel:
            corrupted_img = swap_channels(corrupted_img)
            gt_img = swap_channels(gt_img)
            recons = {k: swap_channels(v) for k, v in recons.items()}

        ssim_corrupted = structural_similarity(corrupted_img, recons['to_corrupted'], multichannel=multichannel)
        ssim_gt = structural_similarity(gt_img, recons['to_gt'], multichannel=multichannel)
        ssim_gt_sm = structural_similarity(gt_img, recons['to_gt_sm'], multichannel=multichannel)

    single_results = dict(psnr_corrupted=psnr_corrupted,
                          psnr_gt=psnr_gt,
                          psnr_gt_sm=psnr_gt_sm,
                          ssim_corrupted=ssim_corrupted,
                          ssim_gt=ssim_gt,
                          ssim_gt_sm=ssim_gt_sm,
                          mse_corrupted=mse_corrupted,
                          mse_gt=mse_gt)

    for key, value in single_results.items():
        if key not in list(results.keys()):
            results[key] = []
        results[key].append(value)

    return results


def track_uncert_sgld(burnin_iter: int,
                      mcmc_iter: int,
                      iter: int,
                      img: Tensor,
                      sgld_imgs: List[Tensor] = [],
                      **kwargs) -> List[Tensor]:

    if iter > burnin_iter and iter % mcmc_iter == 0:
        sgld_imgs.append(img)
    return sgld_imgs


def save_run(results: dict,
             net: nn.Module,
             optimizer: torch.optim.Optimizer,
             net_input: Tensor,
             out_avg: Tensor,
             sgld_imgs: List[Tensor],
             plot: bool = True,
             path: str = 'run'):

    np_to_pil(torch_to_np(out_avg)[:-1]).save("%s/recon_avg.png" % path)

    results['state_dict'] = net.state_dict()
    results['optimizer'] = optimizer.state_dict()
    results['net_input'] = net_input
    results['sgld_imgs'] = sgld_imgs
    torch.save(results, path + '/results.pt')

    if plot:
        np_plot(results, ['mse_corrupted'], labels=[r'corrupted'], ylabel=r'$MSE(d(\hat{\bm{x}}),\tilde{\bm{x}})$', path=path + '/mse_corrupted_losses.png')
        np_plot(results, ['mse_gt'], labels=[r'gt'], ylabel=r'$MSE(\hat{\bm{x}},\bm{x})$', path=path + '/mse_gt_losses.png')

        # np_plot(results, ['elbo'], ylabel=r'$NLL(\hat{\bm{x}},\tilde{\bm{x}})$', path=path + '/losses.png')

        # psnr plot
        labels = ['corrupted', 'gt', 'gt\_sm']
        np_plot(results, ['psnr_corrupted', 'psnr_gt', 'psnr_gt_sm'], labels, r'iteration', r'PSNR', path + '/psnrs.png')

        # ssim plot
        np_plot(results, ['ssim_corrupted', 'ssim_gt', 'ssim_gt_sm'], labels, r'iteration', r'SSIM', path + '/ssims.png')


def get_imgs(img_name: str,
             task: str = 'super-resolution',
             sigma: float = 0.1,
             imsize: Union[Tuple[int], int] = -1,
             factor: int = 4,
             enforce_div32: str = 'CROP'):

    fname = get_fname(img_name)
    if task == 'super_resolution':
        imgs = load_LR_HR_imgs_sr(fname, imsize, factor, enforce_div32)
    elif task == 'denoising':
        img_pil = crop_image(get_image(fname, imsize)[0], d=32)
        img_np = pil_to_np(img_pil)
        img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma)
        imgs = {'gt': img_np, 'noisy': img_noisy_np}
    elif task == 'inpainting':
        img_pil, img_np = get_image(fname, imsize)
        mask_path = fname.split('.')
        mask_path[0] += '_mask'
        mask_path = '.'.join(mask_path)
        img_mask_pil, img_mask_np = get_image(mask_path, imsize)
        imgs = {'gt': img_np, 'mask': img_mask_np}
    return imgs


def get_net_and_optim(num_input_channels: int,
                      num_output_channels: int,
                      num_channels_down: Union[List[int], int],
                      num_channels_up: Union[List[int], int],
                      num_channels_skip: Union[List[int], int],
                      num_scales: int,
                      filter_size_down: int = 3,
                      filter_size_up: int = 3,
                      filter_skip_size: int = 1,
                      downsample_mode: str = 'stride',
                      upsample_mode: str = 'bilinear',
                      pad: str = 'reflection',
                      need1x1_up: bool = True,
                      net_specs: dict = {},
                      optim_specs: dict = None):

    if optim_specs is None:
        optim_specs = dict(lr=0.01)

    # net = get_net(num_input_channels, net_type, pad,
    #               skip_n33d=num_channels_down,
    #               skip_n33u=num_channels_up,
    #               skip_n11=num_channels_skip,
    #               num_scales=num_scales,
    #               n_channels=num_output_channels,
    #               upsample_mode=upsample_mode,
    #               need_sigmoid=False)
    num_channels_down = [num_channels_down] * num_scales if isinstance(num_channels_down, int) else num_channels_down
    num_channels_up = [num_channels_up] * num_scales if isinstance(num_channels_up, int) else num_channels_up
    num_channels_skip = [num_channels_skip] * num_scales if isinstance(num_channels_skip, int) else num_channels_skip

    net = skip(num_input_channels=num_input_channels,
               num_output_channels=num_output_channels,
               num_channels_down=num_channels_down,
               num_channels_up= num_channels_up,
               num_channels_skip=num_channels_skip, filter_size_down=filter_size_down,
               filter_size_up=filter_size_up, filter_skip_size=filter_skip_size,
               pad=pad,
               upsample_mode=upsample_mode,
               downsample_mode=downsample_mode,
               need1x1_up=need1x1_up,
               act_fun="LeakyReLU",
               need_sigmoid=False,
               need_bias=True)

    if 'dropout_type' in list(net_specs.keys()):
        net = MCDropoutVI(net,
                          dropout_type=net_specs['dropout_type'],
                          dropout_p=net_specs['dropout_p'])
        optimizer = torch.optim.AdamW(net.parameters(), **optim_specs)
    elif 'prior_mu' in list(net_specs.keys()):
        prior = {
            'mu': net_specs['prior_mu'],
            'sigma': net_specs['prior_sigma']}
        if 'prior_pi' in list(net_specs.keys()):
            prior['prior_pi'] = net_specs['prior_pi']
        net = MeanFieldVI(net,
                          prior=prior,
                          kl_type=net_specs['kl_type'],
                          beta=net_specs['beta'])
        optimizer = torch.optim.AdamW(net.parameters(), **optim_specs)
    elif 'sgld' in list(net_specs.keys()):
        optimizer = SGLD(net.parameters(), **optim_specs)
    else:
        optimizer = torch.optim.Adam(net.parameters(), **optim_specs)

    return net, optimizer
