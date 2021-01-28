import sys
sys.path.append('..')

from typing import List

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import torch
import torch.nn as nn
import torch.nn.functional as F

from BayTorch.inference.losses import uceloss
from BayTorch.inference.utils import uncert_regression_gal
from BayTorch.visualize import plot_uncert

def np_plot(results: dict,
            keys: List[str],
            labels: List[str] = None,
            xlabel: str = r'iteration',
            ylabel: str = '',
            path: str = 'plot.pdf',
            title: str = '',
            ylim: List[float] = []):

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


def analyse_calibration(net: nn.Module,
                        inputs: torch.tensor,
                        target: torch.tensor,
                        mc_iter: int = 10,
                        post_processor: nn.Module = None,
                        path: str = 'calib.pdf'):

    img_list = []

    with torch.no_grad():
        for _ in range(mc_iter):
            img = net(inputs)
            img[:,:-1] = img[:,:-1].sigmoid()
            img[:,-1:] = torch.exp(-torch.clamp(img[:,-1:], min=-20, max=20))
            if post_processor != None:
                img = post_processor(img)
            img_list.append(img)

    ale, epi, uncert = uncert_regression_gal(img_list, reduction='none')

    out_torch_mean = torch.mean(torch.cat(img_list, dim=0)[:], dim=0, keepdim=True)
    mse_err = F.mse_loss(out_torch_mean[:,:-1], target, reduction='none')
    # if target.size()[1] == 3:
    mse_err = mse_err.mean(dim=1, keepdim=True)

    uce, err_in_bin, avg_sigma_in_bin, freq_in_bin = uceloss(mse_err, uncert)

    fig, _ = plot_uncert(err_in_bin.cpu(), avg_sigma_in_bin.cpu())
    plt.savefig(path)


def analyse_calibration_sgld(img_list: List[torch.Tensor],
                             target: torch.Tensor,
                             criterion: str = 'nll',
                             path: str = 'calib_sgld.pdf'):

    img_list = torch.cat(img_list, dim=0)
    # mean = img_list[:,:-1].mean(dim=0, keepdim=True)
    epi = torch.var(img_list[:,:-1], dim=0, keepdim=True)
    # if epi.size()[1] == 3:
    epi = epi.mean(dim=1, keepdim=True)
    if criterion == 'nll':
        ale = img_list[:,-1:].mean(dim=0, keepdim=True)
    else:
        ale = torch.zeros(epi.size())

    out_torch_mean = torch.mean(img_list[:], dim=0, keepdim=True)
    uncert = ale + epi
    mse_err = F.mse_loss(out_torch_mean[:,:-1], target.cpu(), reduction='none')
    # if target.size()[1] == 3:
    mse_err = mse_err.mean(dim=1, keepdim=True)

    uce, err_in_bin, avg_sigma_in_bin, freq_in_bin = uceloss(mse_err, uncert)

    fig, _ = plot_uncert(err_in_bin.cpu(), avg_sigma_in_bin.cpu())
    plt.savefig(path)
    return ale, epi, uncert, out_torch_mean
