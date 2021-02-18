from typing import Tuple
import time
import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class NLLLoss(nn.Module):

    def __init__(self, reduction: str = 'mean'):
        super(NLLLoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs: Tensor,  target: Tensor) -> Tensor:
        mu = inputs[:,:-1]
        neg_logvar = inputs[:,-1:]
        neg_logvar = torch.clamp(neg_logvar, min=-20, max=20)

        loss = torch.exp(neg_logvar) * torch.pow(target - mu, 2) - neg_logvar

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def add_noise_sgld(model: nn.Module, noise: float):
    for param in [x for x in model.parameters()]:
        n = torch.randn(param.size()) * noise
        n = n.to(param.device)
        param.data = param.data + n


def uncert_regression_gal(img_list: Tensor, reduction: str = 'mean') -> (Tensor):
    img_list = torch.cat(img_list, dim=0)
    mean = img_list[:,:-1].mean(dim=0, keepdim=True)
    ale = img_list[:,-1:].mean(dim=0, keepdim=True)
    epi = torch.var(img_list[:,:-1], dim=0, keepdim=True)
    #if epi.shape[1] == 3:
    epi = epi.mean(dim=1, keepdim=True)
    uncert = ale + epi
    if reduction == 'mean':
        return ale.mean().item(), epi.mean().item(), uncert.mean().item()
    elif reduction == 'sum':
        return ale.sum().item(), epi.sum().item(), uncert.sum().item()
    else:
        return ale.detach(), epi.detach(), uncert.detach()


def uceloss(errors: Tensor,
            uncert: Tensor,
            n_bins: int = 15,
            outlier: float = 0.0,
            range: Tuple[float] = None) -> (Tensor):

    device = errors.device
    if range == None:
        bin_boundaries = torch.linspace(uncert.min().item(), uncert.max().item(), n_bins + 1, device=device)
    else:
        bin_boundaries = torch.linspace(range[0], range[1], n_bins + 1, device=device)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    errors_in_bin_list = []
    avg_uncert_in_bin_list = []
    prop_in_bin_list = []

    uce = torch.zeros(1, device=device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |uncertainty - error| in each bin
        in_bin = uncert.gt(bin_lower.item()) * uncert.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()  # |Bm| / n
        prop_in_bin_list.append(prop_in_bin)
        if prop_in_bin.item() > outlier:
            errors_in_bin = errors[in_bin].float().mean()  # err()
            avg_uncert_in_bin = uncert[in_bin].mean()  # uncert()
            uce += torch.abs(avg_uncert_in_bin - errors_in_bin) * prop_in_bin

            errors_in_bin_list.append(errors_in_bin)
            avg_uncert_in_bin_list.append(avg_uncert_in_bin)

    err_in_bin = torch.tensor(errors_in_bin_list, device=device)
    avg_uncert_in_bin = torch.tensor(avg_uncert_in_bin_list, device=device)
    prop_in_bin = torch.tensor(prop_in_bin_list, device=device)

    return uce, err_in_bin, avg_uncert_in_bin, prop_in_bin

def plot_uncert(err: Tensor,
                sigma: Tensor,
                freq_in_bin: Tensor = None,
                outlier_freq: float = 0.0):

    if freq_in_bin is not None:
        freq_in_bin = freq_in_bin[torch.where(freq_in_bin > outlier_freq)]  # filter out zero frequencies
        err = err[torch.where(freq_in_bin > outlier_freq)]
        sigma = sigma[torch.where(freq_in_bin > outlier_freq)]
    fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.25))
    max_val = np.max([err.max(), sigma.max()])
    min_val = np.min([err.min(), sigma.min()])
    ax.plot([min_val, max_val], [min_val, max_val], 'k--')
    ax.plot(sigma, err, marker='.')
    ax.set_ylabel(r'mse')
    ax.set_xlabel(r'uncertainty')
    ax.set_aspect(1)
    fig.tight_layout()
    return fig, ax
