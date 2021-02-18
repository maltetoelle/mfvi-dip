import sys
from typing import Dict, List, Tuple, Callable, Union
import json

from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.constraints import GreaterThan
from gpytorch.distributions import MultivariateNormal
from gpytorch.priors import GammaPrior, NormalPrior
from gpytorch.likelihoods import GaussianLikelihood, FixedNoiseGaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

import torch
from torch import Tensor
from torch.nn import Module
import torch.multiprocessing as mp

import numpy as np
from scipy.stats import norm
from scipy.signal import find_peaks, find_peaks_cwt
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from tqdm import tqdm

from train_utils import get_net_and_optim
from super_resolution import super_resolution
from denoising import denoising
from inpainting import inpainting


class GPModel(ExactGP):

    def __init__(self,
                 train_x: Tensor,
                 train_y: Tensor,
                 likelihood: GaussianLikelihood,
                 lengthscale_prior: Dict[str, float] = dict(concentration=0.3, rate=1.),
                 lengthscale_constraint: float = 0.01,
                 mean_prior: Dict[str, float] = dict(loc=25., scale=2.)):
        """GP surrogate model for BO.

        Args:
            train_x: training data (n x d)
            train_y: target data (n x 1)
            likeihood: likelihood for GP
            lengthscale_prior: dict. with parameters for alpha prior on lengthscale
            lengthscale_constraint: minimal value for lengthscale
            mean_prior: dict. with parameters for normal prior of mean fn. of GP
        """

        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(
            prior=NormalPrior(**mean_prior)
        )

        lengthscale_prior = GammaPrior(**lengthscale_prior)
        outputscale_prior = GammaPrior(1.0, 1.0)

        self.covar_module = ScaleKernel(
            RBFKernel(
                lengthscale_prior=lengthscale_prior,
                lengthscale_constraint=GreaterThan(lengthscale_constraint)
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
                     num_iter: int = 1000,
                     lengthscale_prior: Dict[str, float] = dict(concentration=0.3, rate=1.),
                     lengthscale_constraint: float = 0.01,
                     mean_prior: Dict[str, float] = dict(loc=25., scale=2.),
                     noise_prior: Union[float, Dict[str, float]] = 1e-4) -> Tuple[ExactGP, GaussianLikelihood]:
    """Fn. for initalizing and training surrogate GP model for BO.

    Args:
        train_x: training data (n x d)
        train_y: target data (n x 1)
        num_iter: number of iterations for training GP
        lengthscale_prior: dict. with parameters for alpha prior on lengthscale
        lengthscale_constraint: minimal value for lengthscale
        mean_prior: dict. with parameters for normal prior of mean fn. of GP
        noise_prior: either dict with parameters for alpha prior for noise
                     or float for constant noise

    Returns:
        GP model and corresponding likelihood
    """

    if isinstance(noise_prior, float):
        likelihood = FixedNoiseGaussianLikelihood(
            noise=torch.ones_like(train_x) * noise_prior
        )
    else:
        likelihood = GaussianLikelihood(
            noise_prior=GammaPrior(**noise_prior)
        )

    model = GPModel(
        train_x, train_y, likelihood, lengthscale_prior,
        lengthscale_constraint, mean_prior
    ).double().to(train_x.device)
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam([{'params': model.parameters()},], lr=0.1)
    mll = ExactMarginalLogLikelihood(likelihood, model)

    print("Training GP...")
    pbar = tqdm(range(num_iter), file=sys.stdout)
    for i in pbar:
        optimizer.zero_grad()
        output = model(train_x.double())
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

        pbar.set_description(f"loss: {loss.item():.3f} | lengthscale: {model.covar_module.base_kernel.lengthscale.item():.3f} | noise: {model.likelihood.noise.mean().item():.5f}")

    return model, likelihood


def expected_improvement(model: ExactGP,
                         likelihood: GaussianLikelihood,
                         params_space: Tensor,
                         params_samples: Tensor,
                         cost_samples: Tensor,
                         xi: float = 0.01) -> Tensor:
    '''
    Computes the EI at points for the parameter space based on
    cost samples using a Gaussian process surrogate model.

    Args:
        model: surrogate GP model
        likeliood: corresponding likelihood
        params_space: Parameter space at which EI shall be computed (m x d).
        params_samples: already evaluated parameters (n x d)
        cost_samples: Sample values at params_samples (n x 1).
        xi: Exploitation-exploration trade-off parameter.

    Returns:
        Expected improvements for paramter space.
    '''
    # make prediction for whole parameter space
    model.eval()
    likelihood.eval()

    device = model.covar_module.base_kernel.lengthscale.device

    with torch.no_grad():
        pred = likelihood(model(torch.from_numpy(params_space).double().to(device)))
        pred_sample = likelihood(model(params_samples.double().to(device)))

    mu, sigma = pred.mean.cpu().numpy(), pred.stddev.cpu().numpy()
    mu_sample = pred_sample.mean.cpu().numpy()

    sigma = sigma.reshape(-1, 1)

    # We have to make sure to not devide by 0
    with np.errstate(divide='warn'):
        # imp = mu - np.max(cost_samples.numpy()) - xi # noise free version
        imp = mu - np.max(mu_sample) - xi
        Z = imp.reshape(-1,1) / sigma
        ei = imp.reshape(-1,1) * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei


def propose_location_multidim(model: ExactGP,
                              likelihood: GaussianLikelihood,
                              eval_acq: Callable,
                              bounds: np.ndarray,
                              n_restarts: int = 25,
                              batch_size: int = 1) -> np.ndarray:
    '''
    Proposes the next sampling point by optimizing the acquisition fct.

    Args:
        model: surrogate GP model
        likelihood: corresponding likelihood
        eval_acq: eval. fn. for acquisition
        bounds: bounds on params (d x 2)
        n_restars: restarts for minimizer to find max of acquisition fn.
        batch_size: number of parameters to propose

    Returns:
        Location of the acquisition function maximums
    '''

    min_val = 1
    min_x = None

    def min_obj(params):
        # Minimization objective is the negative acquisition function
        return - eval_acq(params, model, likelihood).flatten()

    results = []
    # Find the best optimum by starting from n_restart different random points.
    for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, bounds.shape[0])):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
        results.append([res.fun[0], res.x[0]])
        # if res.fun < min_val:
        #     min_val = res.fun[0]
        #     min_x = res.x
    # return min_x.tolist()#.reshape(1, -1)
    results = np.array(results) * -1
    acq_peaks_idx = find_peaks_cwt(results[:,0], np.arange(1, 10))
    acq_peaks = results[:,1][acq_peaks_idx]
    return np.sort(acq_peaks)[-batch_size:].to_list()


def plot_optimization(model: ExactGP,
                      likelihood: GaussianLikelihood,
                      # acq_fn: Callable,
                      acquisition: np.ndarray,
                      next_params: Tensor,
                      params_space: Tensor,
                      params_samples: Tensor,
                      cost_samples: Tensor,
                      path: str = None):
    '''Helper fn. for plotting results during training.

    Args:
        model: surrogate GP model of BO
        likelihood: corresponding likelihood
        acquisition: values of the acquisition fn. on parameter space
        next_params: proposed next parameter
        params_space: whole parameter space
        params_samples: already evaluated parameter samples
        cost_samples: corresponding evaluated costs
        path: where to save plot. None for no saving
    '''

    cp = sns.color_palette()

    model.eval()
    likelihood.eval()

    with torch.no_grad():
        pred = likelihood(model(params_space.double()))#.type(self.dtype)))

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
    sur_fct, = ax.plot(p_space, mu, c=cp[1], label='Surrogate mean')
    cost_samples, = ax.plot(p_samples, c_samples, 'kx', mew=3, label='Cost samples')

    ax2 = ax.twinx()
    # acquisition = acq_fn(params_space.cpu().numpy(), model, likelihood)
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
    ax2.set_yticks(np.round(yt2, 3))

    ax.set_ylabel(r"PSNR$(\bm{x},\hat{\bm{x}})$", fontsize=17)
    ax2.set_ylabel(r"Expected Improvement", fontsize=17)
    ax.set_xlabel(r"$\sigma_p$", fontsize=22)

    # if self.next_params:
    # ax.axvline(x=params_space.cpu().numpy()[np.argmax(acquisition)], ls='--', c='r', zorder=10)
    for p in next_params.flatten().cpu().numpy():
        ax.axvline(x=p, ls='--', c='r', zorder=10)

    ax2.legend(handles=[sur_fct, cost_samples, acq_fct], loc='lower right')

    if path is not None:
        plt.tight_layout()
        plt.savefig(path, bbox_inches='tight')

    plt.show()

def plot_convergence(params_samples: Tensor,
                     cost_samples: Tensor,
                     n_init: int = 0,
                     path: str = None):
    '''Helper fn. to plot convergence after training.

    Args:
        params_samples: already evaluated parameter values
        cost_samples: corresponding evaluated costs
        n_init: number of initial random samples
        path: where to save plot. None for no saving
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
        try:
            plt.tight_layout()
            plt.savefig(path, bbox_inches='tight')
        except:
            plt.savefig(path)#, bbox_inches='tight')

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

import pdb
class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

class BatchedDIPProblem(Module):

    def __init__(self,
                 task: str,
                 config: Dict,
                 metric: str = "psnr_gt",
                 img_name: str = "xray",
                 criterion: str = 'nll',
                 num_iter_eval_fn: int = 10000,
                 save_trials: bool = False,
                 path_log_dir: str = ".",
                 gpus: List[int] = [0, 1],
                 seed: int = None):
        """Class for performing DIP training with multiple Hyperparams
        across multiple GPUs.

        Args:
            task: denoising, super_resolution, inpainting
            config: config for BO
            metric: metric to be evaluated in BO
            img_name: abbreviation of image name
            criterion: loss fn.
            num_iter_eval_fn: number of iterations for DIP run
            save_trials: save DIP runs
            path_log_dir: where to save DIP runs
            gpus: list of GPUs on which to perform training
            seed: seed for reproducibility
        """

        super(BatchedDIPProblem, self).__init__()

        if task == "denoising":
            self.fn = denoising
            self.net_structure = json.load(open("./configs/net_den.json"))
        elif task == "super_resolution":
            self.fn = super_resolution
            self.net_structure = json.load(open("./configs/net_sr.json"))
        elif task == "inpainting":
            self.fn = inpainting
            self.net_structure = json.load(open("./configs/net_inp.json"))

        self.net_params = config["net_params"]
        self.optim_params = config["optim_params"]
        self.net_specs = config["net_specs"]
        self.optim_specs = config["optim_specs"]
        self.metric = metric
        self.criterion=criterion
        self.seed = seed

        self.params = [p["name"] for p in config["parameter"]]

        self.img_name = img_name
        self.num_iter_eval_fn = num_iter_eval_fn
        self.save_trials = save_trials
        self.path_log_dir = path_log_dir
        self.gpus = gpus

        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            print("spawn method has already been set")

    def mp_fn(self,
              gpu: int,
              single_params: Tensor,
              results: Tensor,
              order: int):
        """Fn. for training one DIP run on one GPU.

        Args:
            gpu: integer of GPU
            single_params: hyperparams in net_specs or optim_specs which to optimize
            results: output Tensor of final results (batch_size x 1)
            order: order in which it started to set that row in results tensor after trainig
        """

        device = 'cuda:' + str(gpu)
        net_specs = self.net_specs.copy()
        optim_specs = self.optim_specs.copy()

        for i, param in enumerate(self.params):
            if param in self.optim_params:
                optim_specs[param] = float(single_params[i])
            else:
                net_specs[param] = float(single_params[i])

        net, optimizer = get_net_and_optim(net_specs=net_specs, optim_specs=optim_specs, **self.net_structure)

        single_result = self.fn(img_name=self.img_name,
                                num_iter=self.num_iter_eval_fn,
                                criterion=self.criterion,
                                net_specs=net_specs,
                                optim_specs=optim_specs,
                                save=self.save_trials,
                                path_log_dir=self.path_log_dir,
                                gpu=gpu,
                                net=net,
                                optimizer=optimizer,
                                seed=self.seed)

        res = single_result[self.metric]
        results[order] = torch.tensor([np.max(res)])

        results = results[torch.where(results != 0.)[0]]

    def forward(self, params: Tensor) -> Tensor:
        """Train multiple DIPs on multiple GPUs

        Args:
            params: values of parameters to be optimized
        """
        results = torch.zeros(len(params))
        processes = []
        _gpus = (self.gpus * np.ceil(len(params)/len(self.gpus)).astype('int'))[:len(params)]
        for order, (single_params, gpu) in enumerate(zip(params, _gpus)):
            print(f"Starting process {order+1}/{len(params)}...")
            p = mp.Process(target=self.mp_fn, args=(gpu, single_params, results, order,))
            p.start()
            processes.append(p)
        print('Waiting for processes to finish')
        for p in processes:
            p.join()

        return results
