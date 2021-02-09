from typing import Dict, List, Tuple, Callable, Union

from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.priors import GammaPrior, NormalPrior
from gpytorch.likelihoods import GaussianLikelihood, FixedNoiseGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

import torch
from torch import Tensor
from torch.nn import Module
import torch.multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import lpips

from train_utils import get_net_and_optim
from super_resolution import super_resolution
from denoising import denoising
from inpainting import inpainting


class GPModel(ExactGP):

    def __init__(self,
                 train_x: Tensor,
                 train_y: Tensor,
                 likelihood: Tensor,
                 lengthscale_prior: Dict[str, float] = dict(concentration=0.3, rate=1.),
                 mean_prior: Dict[str, float] = dict(loc=25., scale=2.)):

        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(
            prior=NormalPrior(**mean_prior)
        )

        lengthscale_prior = GammaPrior(**lengthscale_prior)
        outputscale_prior = GammaPrior(1.0, 1.0)

        self.covar_module = ScaleKernel(
            RBFKernel(
                lengthscale_prior=lengthscale_prior,
            ),
            outputscale_prior=outputscale_prior
        )

        self.covar_module.base_kernel.lengthscale = lengthscale_prior.mean
        self.covar_module.outputscale = outputscale_prior.mean

    def forward(self, x: Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


def initialize_model(train_x: Tensor,
                     train_y: Tensor,
                     num_iter: int = 100,
                     lengthscale_prior: Dict[str, float] = dict(concentration=0.3, rate=1.),
                     mean_prior: Dict[str, float] = dict(loc=25., scale=2.),
                     noise_prior: Union[float, Dict[str, float]] = 1e-4) -> Tuple[ExactGP, GaussianLikelihood]:

    if isinstance(noise_prior, float):
        likelihood = FixedNoiseGaussianLikelihood(
            noise=torch.ones_like(train_x) * noise_prior
        )
    else:
        likelihood = GaussianLikelihood(
            noise_prior=GammaPrior(**noise_prior)
        )

    model = GPModel(
        train_x, train_y, likelihood,
        lengthscale_prior, mean_prior
    ).double().to(train_x.device)
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam([{'params': model.parameters()},], lr=0.1)
    mll = ExactMarginalLogLikelihood(likelihood, model)

    for i in range(num_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    return model, likelihood


def plot_optimization(model: ExactGP,
                      likelihood: GaussianLikelihood,
                      acq_fn: Callable,
                      params_space: Tensor,
                      params_samples: Tensor,
                      cost_samples: Tensor,
                      path: str = None):
    '''
    Helper fct. for plotting results during training.
    '''

    cp = sns.color_palette()

    model.eval()
    likelihood.eval()
    with torch.no_grad():
        pred = likelihood(model(params_space.flatten()))#.type(self.dtype)))

    mu = pred.mean.cpu().numpy()
    lower, upper = pred.confidence_region()
    lower, upper = lower.cpu().numpy(), upper.cpu().numpy()

    p_samples = params_samples.cpu().numpy().flatten()
    c_samples = cost_samples.cpu().numpy().flatten()
    p_space = params_space.cpu().numpy().ravel()
    fig, ax = plt.subplots()
    ax.fill_between(p_space,
                    lower,
                    upper,
                    color=cp[0],
                    alpha=0.5)
    sur_fct, = ax.plot(p_space, mu, c=cp[1], label='Surrogate fct')
    cost_samples, = ax.plot(p_samples, c_samples, 'kx', mew=3, label='Cost samples')

    ax2 = ax.twinx()
    acquisition = acq_fn(params_space.cpu().numpy(), model, likelihood)
    acq_fct, = ax2.plot(p_space,
                        acquisition,
                        color=cp[2],
                        label='Acquisition fct', zorder=0)
    ax2.fill_between(p_space, acquisition.ravel(), 0, color=cp[2], zorder=0, alpha=0.5)
    ax2.set_ylim([0, ax2.get_ylim()[1]*4])
    ax2.grid(False)
    # ax2.set_yticks([])

    yt1 = ax.get_yticks()
    yl1 = ax.get_ylim()
    yl2 = ax2.get_ylim()
    offset = (yl1[0] - yt1[0]) / (yl1[1] - yl1[0]) * (yl2[1] - yl2[0])
    step = (yt1[1] - yt1[0]) / (yl1[1] - yl1[0]) * (yl2[1] - yl2[0])
    yt2 = np.arange(yl2[0] + offset, yl2[1], step)
    # yt2 = np.linspace(yl2[0], yl2[1], len(yt1)-2)
    # yt2 += offset
    ax2.set_yticks(yt2)

    ax.set_ylabel(r"PSNR$(\bm{x},\hat{\bm{x}})$")
    ax2.set_ylabel(r"Expected Improvement")
    ax.set_xlabel(r"$\sigma_p$")

    # if self.next_params:
    ax.axvline(x=params_space.cpu().numpy()[np.argmax(acquisition)], ls='--', c='r', zorder=10)

    ax.legend(handles=[sur_fct, cost_samples, acq_fct], loc='upper right')

    if path is not None:

        plt.tight_layout()
        plt.savefig(path, bbox_inches='tight')
    plt.show()

def plot_convergence(params_samples: Tensor,
                     cost_samples: Tensor,
                     n_init: int = 0,
                     path: str = None):
    '''
    Helper fct. to plot convergence after training.
    '''
    x = params_samples.numpy()[n_init:].ravel()
    y = cost_samples.numpy()[n_init:].ravel()
    r = range(1, len(x)+1)

    x_neighbor_dist = [np.abs(a-b) for a, b in zip(x, x[1:])]
    y_max_watermark = np.maximum.accumulate(y)

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
    ax1.plot(r[1:], x_neighbor_dist, 'bo-')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Distance')
    ax1.set_title('Distance between consecutive params\'s')

    ax2.plot(r, y_max_watermark, 'ro-')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Best params')
    ax2.set_title('Value of best selected sample')

    if path is not None:
        plt.tight_layout()
        plt.savefig(path, bbox_inches='tight')

    plt.show()

# def plot_optimization_2d(self):
#     '''
#     Helper fct. for plotting 2d results during training.
#     '''
#
#     cp = sns.color_palette()
#
#     mu, sigma = self.gpr.predict(self.params_space, return_std=True)
#
#     p0, p1 = self.params_space[:,0].reshape(self.no_pts, self.no_pts), self.params_space[:,1].reshape(self.no_pts, self.no_pts)
#
#     mu = mu.reshape(self.no_pts, self.no_pts)
#     sigma = sigma.reshape(self.no_pts, self.no_pts)
#
#     acquisition = self.acquisition(self.gpr, self.params_space, self.cost_samples)
#     acquisition = acquisition.reshape(self.no_pts, self.no_pts)
#
#     fig, (ax_mean, ax_sigma, ax_acq) = plt.subplots(1, 3)
#
#     ax_mean.pcolormesh(p0, p1, mu)
#     ax_sigma.pcolormesh(p0, p1, sigma)
#     ax_acq.pcolormesh(p0,p1, acquisition)
#
#     plt.show()


class BatchedDIPProblem:

    def __init__(self,
                 task: str,
                 config: Dict,
                 metric: str = "psnr_gt",
                 img_name: str = "xray",
                 num_iter_eval_fn: int = 10000,
                 save_trials: bool = False,
                 path_log_dir: str = ".",
                 gpus: List[int] = [0, 1]):

        # super(BatchedDIPProblem, self).__init__()

        if task == "denoising":
            self.fn = denoising
            # self.path_log_dir = "/bo_den/"
        elif task == "super_resolution":
            self.fn = super_resolution
            # self.path_log_dir = "/bo_sr/"
        elif task == "inpainting":
            self.fn = inpainting
        #     self.path_log_dir = "/bo_inp/"
        # self.path_log_dir += task + "_" + "_".join(metrics) + "_" + datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

        self.net_params = config["net_params"]
        self.optim_params = config["optim_params"]
        self.net_specs = config["net_specs"]
        self.optim_specs = config["optim_specs"]
        self.metric = metric

        self.params = [p["name"] for p in config["parameter"]]

        self.img_name = img_name
        self.num_iter_eval_fn = num_iter_eval_fn
        self.save_trials = save_trials
        self.path_log_dir = path_log_dir
        self.gpus = gpus

        # mp.set_start_method('spawn')

    def mp_fn(self,
              gpu: int,
              single_params: Tensor,
              results: Tensor,
              order: int,
              lpips_loss: lpips.LPIPS):

        # TODO: just for denoising here
        num_input_channels = 32
        num_output_channels = 2
        num_channels_down = 128
        num_channels_up = 128
        num_channels_skip = 4
        num_scales = 5
        upsample_mode = 'bilinear'

        device = 'cuda:' + str(gpu)
        net_specs = self.net_specs.copy()
        optim_specs = self.optim_specs.copy()
        for i, param in enumerate(self.params):
            if param in self.optim_params:
                optim_specs[param] = single_params[i]
            else:
                net_specs[param] = single_params[i]
        net, optimizer = get_net_and_optim(num_input_channels, num_output_channels, num_channels_down, num_channels_up, num_channels_skip, num_scales, net_specs=net_specs, optim_specs=optim_specs)

        net = net.to(device)
        lpips_loss = lpips_loss.to(device)

        single_result = self.fn(img_name=self.img_name,
                                num_iter=self.num_iter_eval_fn,
                                net_specs=net_specs,
                                optim_specs=optim_specs,
                                save=self.save_trials,
                                path_log_dir=self.path_log_dir,
                                gpu=gpu,
                                net=net,
                                optimizer=optimizer,
                                lpips_loss=lpips_loss)

        res = single_result[self.metric][-int(0.1*self.num_iter_eval_fn):]
        results[order] = torch.tensor([np.mean(res)])

    def eval(self, params: Tensor) -> Tensor:

        results = torch.empty(len(params))
        processes = []
        _gpus = (self.gpus * np.ceil(len(params)/len(self.gpus)).astype('int'))[:len(params)]
        lpips_loss = lpips.LPIPS(net='alex')
        for order, (single_params, gpu) in enumerate(zip(params, _gpus)):
            print(f"Starting process {order+1}/{len(params)}...")
            p = mp.Process(target=self.mp_fn, args=(gpu, single_params, results, order, lpips_loss,))
            p.start()
            processes.append(p)
        print('Waiting for processes to finish')
        for p in processes:
            p.join()

        return results
