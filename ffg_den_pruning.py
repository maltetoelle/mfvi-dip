import sys
sys.path.append('..')

import os
import fire
import platform
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import json

import numpy as np
from models import *

import torch
import torch.optim
import torch.nn.functional as F

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from utils.denoising_utils import *
from utils.bayesian_utils import get_beta_dip
from training import DIPTrainer

from BayTorch import MeanFieldVI, MCDropoutVI
from BayTorch.inference.losses import NLLLoss2d
from BayTorch.inference.utils import prune_weights_ffg_on_the_fly

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

# reg_noise_std = 1./10. # 1./30.
exp_weight = 0.99

def denoising(
            img_name: str = 'xray',
            #bay_inf_type: str = 'mean_field', # 'mc_dropout'
            #criterion: str= 'nll', # 'mse'
            sigma: float = 0.1,
            lr: float = 0.01,
            #weight_decay: float = 1e-4,
            num_iter: int = 50000,
            prune_iter: int = 1000,
            prune_percentage: float = 5.,
            psnr_pruning_thresh: int = 2,
            #dropout_type: float = '2d',
            #dropout_p: float = 0.3,
            prior_mu: float = 0.,
            prior_sigma: float = 0.1,
            prior_pi: list = None,
            kl_type: str = 'reverse',
            beta_type: float = 1e-6,
            reg_noise_std: float = 1./10., # 1./30.
            need_lr_scheduler: bool = False,
            gamma: float = 0.9996,
            burnin_iter: int = 1000,
            analyse_interval: int = 1000,
            gpu: int = 1,
            seed: int = 42
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # path_log_dir = '/home/toelle/logs/%s_den_%s' % (bay_inf_type, datetime.now().strftime("%m_%d_%Y_%H_%M_%S"))
    path_log_dir = '/media/fastdata/toelle/logs/pruning_den_%s' % (datetime.now().strftime("%m_%d_%Y_%H_%M_%S"))

    _sigma = sigma * 255

    if img_name == 'xray':
        fname = 'data/bayesian/BACTERIA-1351146-0006.jpg'
    elif img_name == 'oct':
        fname = 'data/bayesian/CNV-9997680-30.png'
    elif img_name == 'us':
        fname = 'data/bayesian/081_HC.jpg'
    elif img_name == 'ct':
        fname = 'data/bayesian/gt_ct.png'
    elif img_name == 'mri':
        fname = 'data/bayesian/gt_mri.png'
    elif img_name == 'peppers':
        fname = 'GP_DIP/data/denoising/Dataset/image_Peppers512rgb.png'

    img_pil = crop_image(get_image(fname, imsize)[0], d=32)
    img_np = pil_to_np(img_pil)
    img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma)

    num_output_channels = img_np.shape[0] + 1

    if img_np.shape[0] == 3:
        multichannel = True
        _img_np, _img_noisy_np = np.moveaxis(img_np, 0, -1), np.moveaxis(img_noisy_np, 0, -1)
    else:
        multichannel = False
        _img_np, _img_noisy_np = img_np[0], img_noisy_np[0]

    net_info = {
        # 'bay_inf_type': bay_inf_type,
        # 'criterion': criterion,
        'num_input_channels': num_input_channels,
        'num_channels_down': num_channels_down,
        'num_channels_up': num_channels_up,
        'num_channels_skip': num_channels_skip,
        'num_output_channels': num_output_channels,
        'num_scales': num_scales,
        'upsample_mode': upsample_mode,
        # 'dropout_type': dropout_type,
        # 'dropout_p': dropout_p,
        'prior_mu': prior_mu,
        'prior_sigma': prior_sigma,
        'prior_pi': prior_pi,
        'kl_type': kl_type,
        'beta_type': beta_type,
        'need_lr_scheduler': need_lr_scheduler,
        'gamma': gamma,
        # 'burnin_iter': burnin_iter,
        'seed': seed,
        'lr': lr,
        # 'weight_decay': weight_decay
    }

    if platform.system() == 'Linux':
        os.mkdir(path_log_dir)
        img_noisy_pil.save("%s/noisy_img.png" % path_log_dir)
        with open(path_log_dir + '/net_info.json', 'w') as f:
            json.dump(net_info, f, indent=4)

    net = get_net(num_input_channels, 'skip', 'reflection',
                  skip_n33d=num_channels_down,
                  skip_n33u=num_channels_up,
                  skip_n11=num_channels_skip,
                  num_scales=num_scales,
                  n_channels=num_output_channels,
                  need_sigmoid=False,
                  upsample_mode='bilinear').type(dtype)

    trainer = DIPTrainer('mean_field')

    # if bay_inf_type == 'mc_dropout':
    #     net = MCDropoutVI(net, dropout_type=dropout_type, dropout_p=dropout_p, output_dip_drop=False)
    #     net.apply(init_normal)
    #     optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    #
    # elif bay_inf_type == 'mean_field':
    prior = {'mu': prior_mu, 'sigma': prior_sigma}
    if prior_pi is not None:
        prior['pi'] = prior_pi
    net = MeanFieldVI(net, prior=prior, kl_type=kl_type, _version='new').type(dtype)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)

    net_input = get_noise(num_input_channels, 'noise', (img_pil.size[1], img_pil.size[0])).type(dtype).detach()

    # Compute number of parameters
    s  = sum([np.prod(list(p.size())) for p in net.parameters()]);
    print ('Number of params: %d' % s)

    # Loss
    _criterion = NLLLoss2d(reduction='mean').type(dtype)

    img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)

    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    out_avg = None

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    for i in range(num_iter):
        optimizer.zero_grad()

        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        out = net(net_input)
        out[:,:-1] = torch.sigmoid(out[:,:-1])

        # Smoothing
        if out_avg is None:
            out_avg = out.detach()
        else:
            out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

        # if criterion == 'nll':
        nll = _criterion(out, img_noisy_torch)
        # else:
        #     nll = _criterion(out[:,:-1], img_noisy_torch)

        # if bay_inf_type == 'mean_field':
        beta = get_beta_dip(i, num_iter, beta_type, net)
        kl = net.kl.type(dtype) * beta
        # if beta_type == 'no_params':
        #     nll *= torch.tensor(img_noisy_torch.size()).prod()
        ELBOloss = nll + kl
        # else:
        #     kl = torch.tensor([0])
        #     ELBOloss = nll

        ELBOloss.backward()
        optimizer.step()

        if need_lr_scheduler:
            lr_scheduler.step()

        out_np = out.detach().cpu().numpy()[0]
        out_avg_np = out_avg.detach().cpu().numpy()[0]

        psnr_noisy = peak_signal_noise_ratio(img_noisy_np, out_np[:-1])
        psnr_gt    = peak_signal_noise_ratio(img_np, out_np[:-1])
        psnr_gt_sm = peak_signal_noise_ratio(img_np, out_avg_np[:-1])

        with torch.no_grad():
            mse = F.mse_loss(out[:,:-1], img_noisy_torch)

        if multichannel:
            _out_np = np.moveaxis(out_np[:-1], 0, -1)
            _out_avg_np = np.moveaxis(out_avg_np[:-1], 0, -1)
        else:
            _out_np = out_np[0]
            _out_avg_np = out_avg_np[0]

        ssim_noisy = structural_similarity(_img_noisy_np, _out_np, multichannel=multichannel)
        ssim_gt    = structural_similarity(_img_np, _out_np, multichannel=multichannel)
        ssim_gt_sm = structural_similarity(_img_np, _out_avg_np, multichannel=multichannel)

        if i % analyse_interval == 0:
            trainer.analyse_calibration(net, net_input, img_noisy_torch)
            # trainer.track_param_dist(net)

        if i > burnin_iter and i % prune_iter == 0 and i != 0:
            _prune_percentage = prune_percentage / 100

            psnr_noisy_pruning = psnr_noisy
            total_amount = 0.

            _net = nn.Sequential(net._modules.copy())
            _net.load_state_dict(net.state_dict().copy())

            while (psnr_noisy - psnr_noisy_pruning) < psnr_pruning_thresh and total_amount < prune_percentage:

                prune_weights_ffg_on_the_fly(_net, amount=0.01)
                total_amount += 1

                with torch.no_grad():
                    outs_prune = []
                    for _ in range(10):
                        out_prune = net(net_input)
                        out_prune[:,:-1] = torch.sigmoid(out[:,:-1])
                        outs_prune.append(torch_to_np(out_prune)[:-1])
                    out_np_prune = np.mean(np.array(outs_prune), axis=0)
                    psnr_noisy_pruning = peak_signal_noise_ratio(img_noisy_np, out_np_prune)
                    print(psnr_noisy_pruning)

                if (psnr_noisy - psnr_noisy_pruning) < psnr_pruning_thresh:
                    prune_weights_ffg_on_the_fly(net, amount=0.01)

            del _net

        trainer.track_training(kl=kl, nll=nll, elbo=ELBOloss, mse=mse, psnr_noisy=psnr_noisy, psnr_gt=psnr_gt, psnr_gt_sm=psnr_gt_sm, ssim_noisy=ssim_noisy, ssim_gt=ssim_gt, ssim_gt_sm=ssim_gt_sm)

        print('I: %d | ELBO: %.2f | NLL: %.2f | KL: %.2f | MSE: %.2f | PSNR_noisy: %.2f | PSNR_gt: %.2f | PSNR_gt_sm: %.2f' % (i, ELBOloss.item(), nll.item(), kl.item(), mse.item(), psnr_noisy, psnr_gt, psnr_gt_sm), '\r', end='')

    np_to_pil(out_avg_np[:-1]).save("%s/recon_avg.png" % path_log_dir)

    trainer.save(net, optimizer, path_log_dir + '/train_vals.pt')

    trainer.np_plot(['mse'], ylabel=r'$MSE(\hat{\bm{x}},\tilde{\bm{x}})$', path=path_log_dir + '/mse_losses.png')
    trainer.np_plot(['nll', 'kl', 'elbo'], ylabel=r'$NLL(\hat{\bm{x}},\tilde{\bm{x}})$', path=path_log_dir + '/losses.png')
    trainer.np_plot(['ale', 'epi', 'uncert'], labels=['Aleatoric', 'Epistemic', 'Total'], ylabel=r'uncertainty', path=path_log_dir + '/uncertainties.png')

    # psnr plot
    labels = ['noisy', 'gt', 'gt\_sm']
    trainer.np_plot(['psnr_noisy', 'psnr_gt', 'psnr_gt_sm'], labels, r'iteration', r'PSNR', path_log_dir + '/psnrs.png')

    # ssim plot
    trainer.np_plot(['ssim_noisy', 'ssim_gt', 'ssim_gt_sm'], labels, r'iteration', r'SSIM', path_log_dir + '/ssims.png')

    # trainer.plot_weight_dist(net, path_log_dir + '/weight_dist.png')
    # trainer.plot_log_weight_dist(net, path_log_dir + '/log_weight_dist.png')

    np.savez(path_log_dir + '/vals.npz', net_input=net_input.cpu().numpy(), img_name=img_name, sigma=sigma)

    for i in range(5):
        trainer.analyse_calibration(net, net_input, img_noisy_torch, mc_iter=mc_iter, plot=True, path=path_log_dir + '/calibs_%d.png' % i)

if __name__ == '__main__':
    fire.Fire(denoising)
