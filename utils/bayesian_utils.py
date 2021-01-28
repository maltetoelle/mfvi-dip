import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from decimal import Decimal
from .common_utils import torch_to_np, np_to_torch

def add_noise(model, lr):
    for n in [x for x in model.parameters() if len(x.size()) == 4]:
        noise = torch.randn(n.size()) * lr
        noise = noise.to(n.device)
        n.data = n.data + noise

def calc_uncert(img_list, noisy_img, reduction='mean'):
    img_list = torch.cat(img_list, dim=0)
    ale = img_list[:,-1:].mean(dim=0, keepdim=True)
    epi = torch.var(img_list[:,:-1], dim=0, keepdim=True)
    if epi.shape[1] == 3:
        epi = epi.mean(dim=1, keepdim=True)
    uncert = ale + epi
    if reduction == 'mean':
        return ale.mean().item(), epi.mean().item(), uncert.mean().item()
    else:
        return ale.detach(), epi.detach(), uncert.detach()
    # return ale.exp(), epi, uncert

# Loss
def gaussian_nll(mu, neg_logvar, target, reduction: str = 'mean', weight_decay: float = 0.0, model: nn.Module = nn.Sequential()):
    neg_logvar = torch.clamp(neg_logvar, min=-20, max=20)
    loss = torch.exp(neg_logvar) * torch.pow(target - mu, 2) - neg_logvar

    #loss = torch.exp(-logvar) * torch.pow(target - mu, 2) + logvar
    #loss = -logvar**2 * torch.pow(target-mu, 2) + torch.clamp(torch.log(logvar + 0.001), 0, 1)

    if reduction == 'mean':
        return loss.mean() + _weight_decay(model, weight_decay)#, _weight_decay(model, weight_decay)
    elif reduction == 'sum':
        return loss.sum() + _weight_decay(model, weight_decay)
    else:
        return loss
    # return loss.mean() if reduction == 'mean' else loss.sum()

def uceloss(errors, uncert, n_bins=15, outlier=0.0, range=None):
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

def plot_uncert(err, sigma, freq_in_bin=None, outlier_freq=0.0):
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

class Hook():
    def __init__(self, module, backward=False):
        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()

class ELBO(nn.Module):
    def __init__(self):
        super(ELBO, self).__init__()

    def forward(self, output, target, kl, beta: float = 1.):
        assert not target.requires_grad
        nll_loss = gaussian_nll(output[:,:-1], output[:,-1:], target, reduction='mean')
        return nll_loss + beta * kl

def kl_divergence(mu_q, sig_q, mu_p, sig_p):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl

def get_beta_dip(iter, num_iter, beta_type, net):
    if type(beta_type) is float:
        return beta_type

    if beta_type == "Blundell":
        beta = 2 ** Decimal((num_iter - (iter + 1))) / (2 ** Decimal(num_iter) - 1)
    elif beta_type == "Soenderby":
        #if epoch is None or num_epochs is None:
        #    raise ValueError('Soenderby method requires both epoch and num_epochs to be passed.')
        beta = min(iter / (num_iter // 4), 1)
    elif beta_type == "Standard":
        beta = 1 / num_iter
    elif beta_type == 'Linear':
        beta = 1 - (1 / num_iter) * iter
    elif beta_type == 'no_params':
        no_params  = sum(np.prod(list(p.size())) for p in net.parameters())
        beta = 1 / no_params
    else:
        beta = 0
    return float(beta)
