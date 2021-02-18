import sys
sys.path.append('..')

from typing import Union, List, Callable, Tuple, Dict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from utils.common_utils import torch_to_np, np_to_pil, pil_to_np, get_fname, get_image, crop_image, init_normal
from utils.denoising_utils import get_noisy_image
from utils.sr_utils import load_LR_HR_imgs_sr
from utils.bayesian_utils import NLLLoss

from models import skip
from models.vi import MeanFieldVI, MCDropoutVI

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{bm}'


def peak_signal_noise_ratio(image_true: Tensor, image_test: Tensor):
    """Compute PSNR on GPU.
    We always assume float images with a max. value of 1.

    Args:
        image_true: ground truth image, same shape as test image
        image_test: test image
    """
    err = F.mse_loss(image_true, image_test)
    return 10 * torch.log10(1 / err)


def np_plot(results: dict,
            keys: List[str],
            labels: List[str] = None,
            xlabel: str = r'iteration',
            ylabel: str = '',
            path: str = 'plot.pdf',
            title: str = '',
            ylim: List[float] = []):
    """Convenience fn. for plotting final training results.

    Args:
        results: dcit. storing all training stats
        keys: keys for accessing needed information of results
        labels: set labels different to keys
        xlabel: label for x-axis
        ylabel: label for y-axis
        path: where to save plot
        title: title for plot
        ylim: set limit for y-axis
    """

    if labels is None:
        labels = keys
    fig, ax = plt.subplots(1,1)
    for key, label in zip(keys, labels):
        ys = results[key]
        ax.plot(range(len(ys)), ys, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.legend()
    plt.savefig(path)


def closure(net: nn.Module,
            optimizer: torch.optim.Optimizer,
            inputs: Tensor,
            target: Tensor,
            criterion: nn.Module,
            post_processor: nn.Module = None,
            mask: Tensor = torch.ones(1)):
    """One optimization step.

    Args:
        net: DIP model (can be standard, MFVI or MCDropout)
        optimizer: optimizer to use
        inputs: input to net
        target: ground truth
        criterion: loss fn.
        post_processor: process output of net before computing loss (e.g. downsampler in SR)
        mask: multiply output and target by mask before computing loss (for inpainting)
    """

    mask = mask.to(inputs.device)
    optimizer.zero_grad()

    out = net(inputs)
    out[:,:-1] = torch.sigmoid(out[:,:-1])

    if post_processor is not None:
        out_p = post_processor(out)
    else:
        out_p = out

    if isinstance(criterion, NLLLoss):
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


def get_mc_preds(net: nn.Module,
                 inputs: Tensor,
                 mc_iter: int = 25,
                 post_processor: nn.Module = None,
                 mask: Tensor = torch.ones(1)) -> List[Tensor]:
    """Convenience fn. for MC integration for uncertainty estimation.

    Args:
        net: DIP model (can be standard, MFVI or MCDropout)
        inputs: input to net
        mc_iter: number of MC samples
        post_processor: process output of net before computing loss (e.g. downsampler in SR)
        mask: multiply output and target by mask before computing loss (for inpainting)
    """
    img_list = []
    with torch.no_grad():
        for _ in range(mc_iter):
            out = net(inputs)
            if post_processor is not None:
                out = post_processor(out)
            out *= mask.to(out.device)
            out[:,:-1] = torch.sigmoid(out[:,:-1])
            out[:,-1:] = torch.exp(-torch.clamp(out[:,-1:], min=-20, max=20))
            img_list.append(out)
    return img_list


def track_training(corrupted_img: Tensor,
                   gt_img: Tensor,
                   recons: Dict[str, Tensor],
                   results: Dict[str, List] = dict()):
    """Convenience fn. for tracking all important stats during training.

    Args:
        corrupted_img: degraded image (e.g. noisy, low-resolution)
        gt_img: undegraded, ground truth image
        recons: Dictionary with key (to corrupted, to_gt, to_gt_sm) indicates
                which measure shall be computed
        results: Dictionary storing all former training results
    """
    with torch.no_grad():
        mse_corrupted = F.mse_loss(recons['to_corrupted'][:,:-1], corrupted_img).item()
        mse_gt = F.mse_loss(recons['to_gt'][:,:-1], gt_img).item()

        psnr_corrupted = peak_signal_noise_ratio(corrupted_img, recons['to_corrupted'][:,:-1]).item()
        psnr_gt = peak_signal_noise_ratio(gt_img, recons['to_gt'][:,:-1]).item()
        psnr_gt_sm = peak_signal_noise_ratio(gt_img, recons['to_gt_sm'][:,:-1]).item()

    single_results = dict(psnr_corrupted=psnr_corrupted,
                          psnr_gt=psnr_gt,
                          psnr_gt_sm=psnr_gt_sm,
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
    """Convenience fn. for storing SGLD uncertainty images.

    Args:
        burnin_iter: burn in iterations for the Markov Chain
        mcmc_iter: iterations between consecutive samples.
                   Needed because of dependence of weight samples
        img: recon to store
        sgld_imgs: list holding all former imgs
    """
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
    """Convenience fn. for saving one run with the required plots for quick examination.

    Args:
        results: dict. storing results of run
        net: DIP model
        optimizer: used optimizer
        net_input: input do net sampled from uniform or normal distribution
        out_avg: exponential smoothed image
        sgld_imgs: used for uncertainty quantification for SGLD
        plot: Wether to plot results
        path: where to save results
    """

    np_to_pil(torch_to_np(out_avg)[:-1]).save("%s/recon_avg.png" % path)

    results['state_dict'] = net.state_dict()
    results['optimizer'] = optimizer.state_dict()
    results['net_input'] = net_input
    results['sgld_imgs'] = sgld_imgs
    torch.save(results, path + '/results.pt')

    if plot:
        np_plot(results, ['mse_corrupted'], labels=[r'corrupted'], ylabel=r'$MSE(d(\hat{\bm{x}}),\tilde{\bm{x}})$', path=path + '/mse_corrupted_losses.png')
        np_plot(results, ['mse_gt'], labels=[r'gt'], ylabel=r'$MSE(\hat{\bm{x}},\bm{x})$', path=path + '/mse_gt_losses.png')

        # psnr plot
        labels = ['corrupted', 'gt', 'gt\_sm']
        np_plot(results, ['psnr_corrupted', 'psnr_gt', 'psnr_gt_sm'], labels, r'iteration', r'PSNR', path + '/psnrs.png')


def get_imgs(img_name: str,
             task: str = 'super-resolution',
             sigma: float = 0.1,
             domain: str = 'xray',
             imsize: Union[Tuple[int], int] = -1,
             factor: int = 4,
             enforce_div32: str = 'CROP'):
    """Convenience fn. for getting task specifics images.

    Args:
        img_name: abbreviation for image
        sigma: standard deviation for noise in denoising
        domain: domain for noise model in denoising
        imsize: size for images, -1 for no resizing
        factor: downsampling factor for SR
        enforce_div32: images must be divisible by 32
    """

    fname = get_fname(img_name)
    if task == 'super_resolution':
        imgs = load_LR_HR_imgs_sr(fname, imsize, factor, enforce_div32)
    elif task == 'denoising':
        img_pil = crop_image(get_image(fname, imsize)[0], d=32)
        img_np = pil_to_np(img_pil)
        img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma, domain)
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
                      num_scales: int = 5,
                      filter_size_down: int = 3,
                      filter_size_up: int = 3,
                      filter_skip_size: int = 1,
                      downsample_mode: str = 'stride',
                      upsample_mode: str = 'bilinear',
                      pad: str = 'reflection',
                      need1x1_up: bool = True,
                      net_specs: dict = {},
                      optim_specs: dict = None):
    """Convenience fn. for getting DIP model (net) and optimizer for each task.

    Args:
        num_input_channels: input dimensionality
        num_output_channels: output dimensionality
        num_channels_down: channels for encoder, int only possible with num_scales
        num_channels_up: --"-- decoder --"--
        num_channels_skip: --"-- skip connections --"--
        num_scales: number of encoding and decoding operations, needed if int for num_channels_*
        filter_size_down: convolutional kernel size in encoder
        filter_size_up: --"-- decoder
        filter_skip_size: --"-- skip connections
        downsample_mode: mode used for downsampling the input inbetween convolutions of encoder
        upsample_mode: upsample mode for inbetween convolutions of decoder
        pad: padding to input of convolutions
        need1x1_up: additional 1x1 convolutions after every "usual" convolution in decoder
        net_specs: specifications for net (e.g. MFVI, MCDropout, SGLD or conventional DIP)
        optim_specs: specifications for optimizer
    """

    if optim_specs is None:
        optim_specs = dict(lr=0.01)

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
        net.apply(init_normal)
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
    else:
        optimizer = torch.optim.Adam(net.parameters(), **optim_specs)

    return net, optimizer
