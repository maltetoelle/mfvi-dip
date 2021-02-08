from typing import Union, Callable, Dict
from itertools import product
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
sns.set()

from scipy.stats import norm
from scipy.optimize import minimize

import torch
from torch import Tensor
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood

from utils.bo_utils import GPModel, initialize_model, plot_optimization, plot_convergence

class BayesianOptimization:
    '''
    Class for performing Bayesian optimization.
    For now only "expected improvement" is possible as acquisition fct.
    But the code can easily extended to other fct.'s

    Args:
        kernel: kernel for GP regression (is multiplied with constant kernel)
        kernel_params: kernel parameters such as lengthscale in format {'length_scale': 0.01}
        bounds: bounds for the different parameters that shall be optimized
        obj_fn: objective function that sample the value of the function which to optimize
        acquisition: Either str for using an already implemented fct.
                     Or self defined fct. (Callable)
        n_init: number of initial points to sample from objective function
    '''

    def __init__(self,
                 obj_fn: Callable,
                 params: Dict[str, np.ndarray],
                 n_init: int = 1,
                 initial_params_vals: Dict[str, np.ndarray] = None,
                 acq_fn: Union[Callable, str] = 'expected_improvement',
                 acq_kwargs: Dict[str, float] = {}):

        self.obj_fn = obj_fn

        self.params = OrderedDict(params)

        self.n_init = n_init
        self.no_pts = 100

        # initialize whole parameter space for GP regression
        self.bounds = np.array(list(params.values()))
        self.params_space = np.linspace(self.bounds[:,0], self.bounds[:,1], self.no_pts)
        self.params_space = np.array(list(product(*self.params_space.T)))

        self.params_space = torch.from_numpy(self.params_space.astype(np.float32))

        self.params_samples = torch.tensor([])
        self.cost_samples = torch.tensor([])

        if initial_params_vals is not None:
            for i in range(len(initial_params_vals[list(initial_params_vals.keys())[0]])):
                self.params_samples = torch.cat([
                    self.params_samples, torch.tensor([vals[i] for vals in initial_params_vals.values()])
                ], dim=0)
                cost_sample = self.obj_fn({name: val[i] for name, val in initial_params_vals.items()})
                # initial_params_vals = torch.tensor(list(initial_params_vals.values()))
                # TODO: reshape for more than 1 param
                # cost_sample = self.obj_fn(initial_params_vals.reshape(-1, len(self.params)))
                self.cost_samples = torch.cat([self.cost_samples, torch.tensor(cost_sample)], dim=0)
        else:
            for i in range(n_init):
                params_sample = torch.tensor([np.random.uniform(bounds[0], bounds[1]) for bounds in self.params.values()])
                self.params_samples = torch.cat([self.params_samples, params_sample], dim=0)
                cost_sample = self.obj_fn(self.dictionarize(params_sample))
                # cost_sample = self.obj_fn(params_sample)
                self.cost_samples = torch.cat([self.cost_samples, torch.tensor(cost_sample)], dim=0)

        # other acquisition functions can be added here
        if type(acq_fn) == str:
            if acq_fn == 'expected_improvement':
                # TODO: move expected improvement out of the class
                self.acq_fn = self.expected_improvement

        self.eval_acq = lambda params, model, likelihood: self.acq_fn(model, likelihood, params.reshape(-1, len(self.params)), self.params_samples, self.cost_samples, **acq_kwargs)


    def optimize(self,
                 trials: int = 10,
                 # n_restarts: int = 25,
                 num_iter_gp: int = 100,
                 plot: bool = False,
                 gpu: int = 0,
                 path: str = None,
                 lengthscale_prior: Dict[str, float] = dict(concentration=0.3, rate=1.),
                 mean_prior: Dict[str, float] = dict(loc=25., scale=2.),
                 fixed_noise: float = 1e-4) -> Tensor:
        '''
        Fct. for performing Bayesian optimization.

        Args:
            n_iter: optimization budget i.e. how many optimizations to perform
            n_restarts: restarts for minimizer to find max of acquisition fct.
            plot: wether to plot results during training

        Returns:
            Best parameters
        '''
        device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'
        results = {}

        model, likelihood = initialize_model(
            self.params_samples.flatten().to(device), self.cost_samples.flatten().to(device),
            num_iter_gp, lengthscale_prior, mean_prior, fixed_noise
        )

        if path is not None:
            results["p_space"] = self.params_space
            results[0] = {
                "state_dict": model.state_dict(),
                "p_samples": self.params_samples,
                "c_samples": self.cost_samples
            }

        for i in range(1, trials+1):

            # Obtain next sampling point from the acquisition fct.
            # needed for 2D?
            # next_params = self.propose_location(n_restarts)
            acquisition = self.eval_acq(self.params_space.numpy(), model, likelihood)
            next_params = self.params_space[np.argmax(acquisition)]

            # Obtain next noisy sample from the objective fct.
            next_cost = self.obj_fn(self.dictionarize(next_params))
            # next_cost = self.obj_fn(next_params)

            if plot:
                _path = f"{path}/acq_plot_{i}.pdf" if path is not None else None
                if len(self.params) == 1:
                    plot_optimization(
                        model, likelihood, self.eval_acq, self.params_space.to(device),
                        self.params_samples, self.cost_samples, _path
                    )
                # elif len(self.params) == 2:
                #     self.plot_optimization_2d(path)
                else:
                    print('Plotting just up to 2 dimensions possible.')

            self.params_samples = torch.cat([self.params_samples, torch.tensor(next_params)], dim=0)
            self.cost_samples = torch.cat([self.cost_samples, torch.tensor(next_cost)], dim=0)

            model, likelihood = initialize_model(
                self.params_samples.flatten().to(device), self.cost_samples.flatten().to(device),
                num_iter_gp, lengthscale_prior, mean_prior, fixed_noise
            )

            if path is not None:
                results[i] = {
                    "state_dict": model.state_dict(),
                    "p_samples": self.params_samples,
                    "c_samples": self.cost_samples
                }
                torch.save(results, f"{path}/results.pt")

        if plot:
            _path = f"{path}/convergence_plot.pdf" if path is not None else None
            plot_convergence(self.params_samples, self.cost_samples, self.n_init, _path)

        return self.params_samples[np.argmin(self.cost_samples)]


    def propose_location(self, n_restarts: int = 25) -> np.ndarray:
        '''
        Proposes the next sampling point by optimizing the acquisition fct.

        Args:
            n_restars: restarts for minimizer to find max of acquisition fct.

        Returns:
            Location of the acquisition function maximum
        '''

        min_val = 1
        min_x = None

        def min_obj(params):
            # Minimization objective is the negative acquisition function
            return - self.eval_acq(params).flatten()

        # Find the best optimum by starting from n_restart different random points.
        for x0 in np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(n_restarts, self.bounds.shape[0])):
            res = minimize(min_obj, x0=x0, bounds=self.bounds, method='L-BFGS-B')
            if res.fun < min_val:
                min_val = res.fun[0]
                min_x = res.x

        return min_x.tolist()#.reshape(1, -1)

    # @staticmethod
    # def local_min(ys: Tensor) -> Tensor:
    #     return [y for i, y in enumerate(ys)
    #             if ((i == 0) or (ys[i - 1] >= y))
    #             and ((i == len(ys) - 1) or (y < ys[i+1]))]

    @staticmethod
    def expected_improvement(model: ExactGP,
                             # gpr,
                             likelihood: GaussianLikelihood,
                             params_space: Tensor,
                             params_samples: Tensor,
                             cost_samples: Tensor,
                             xi: float = 0.01) -> Tensor:
        '''
        Computes the EI at points for the parameter space based on
        cost samples using a Gaussian process surrogate model.

        Args:
            gpr: A GaussianProcessRegressor fitted to samples.
            params_space: Parameter space at which EI shall be computed (m x d).
            cost_samples: Sample values (n x 1).
            xi: Exploitation-exploration trade-off parameter.

        Returns:
            Expected improvements for paramter space.
        '''
        # make prediction for whole parameter space
        model.eval()
        likelihood.eval()

        dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        with torch.no_grad():
            pred = likelihood(model(torch.from_numpy(params_space).flatten().type(dtype)))
            pred_sample = likelihood(model(params_samples.type(dtype)))

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

    def dictionarize(self, params: np.ndarray) -> Dict[str, float]:
        return {name: val for name, val in zip(self.params.keys(), params)}
