from typing import Union, Callable, Dict
from itertools import product
from collections import OrderedDict

import numpy as np

from scipy.signal import find_peaks, find_peaks_cwt

import torch
from torch import Tensor
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood

from utils.bo_utils import GPModel, initialize_model, plot_optimization, plot_convergence, expected_improvement, propose_location_multidim


class BayesianOptimization:
    '''Class for performing Bayesian optimization.
    For now only "expected improvement" is possible as acquisition fn.
    But the code can easily extended to other fn.'s

    Args:
        obj_fn: objective function that takes as input a Tensor
                of params batch_size x param_dimensionality
        params: dict. of param_name with correspoding bounds
                {param_name: [lower_bound, upper_bound]}
        initial_params_vals: intial parameter values to start BO from
        n_init: if no initial_params_vals defined, draw initial samples
                uniform from search space defined by parameter bounds
        acq_fn: Either str for using an already implemented fn. (e.g. "expected_improvement")
                or self defined fn. (Callable)
                acq_fn(model, likelihood, params, params_samples, cost_samples, **acq_kwargs)
        acq_kwargs: additional kwargs for acq_fn
    '''

    def __init__(self,
                 obj_fn: Callable,
                 params: Dict[str, np.ndarray],
                 initial_params_vals: Dict[str, np.ndarray] = None,
                 n_init: int = 1,
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

        # self.params_samples = torch.tensor([[]])
        self.cost_samples = torch.tensor([])

        if initial_params_vals is not None:
            initial_params_vals = torch.tensor(list(initial_params_vals.values())).reshape(-1, 1)
            # TODO: reshape for more than 1 param
            self.params_samples = initial_params_vals
            cost_sample = self.obj_fn(initial_params_vals.reshape(-1, len(self.params)))
            self.cost_samples = torch.cat([self.cost_samples, torch.tensor(cost_sample)], dim=0)

        else:
            for i in range(n_init):
                params_sample = torch.tensor([[np.random.uniform(bounds[0], bounds[1]) for bounds in self.params.values()]])
                if i == 0:
                    self.params_samples = params_sample
                else:
                    self.params_samples = torch.cat([
                        self.params_samples, params_sample
                    ], dim=0)
                # cost_sample = self.obj_fn(self.dictionarize(params_sample))

                cost_sample = self.obj_fn(params_sample)
                self.cost_samples = torch.cat([self.cost_samples, torch.tensor(cost_sample)], dim=0)

        self.params_samples = self.params_samples.reshape((self.cost_samples.size(0), -1))

        # other acquisition functions can be added here
        if type(acq_fn) == str:
            if acq_fn == 'expected_improvement':
                self.acq_fn = expected_improvement
        else:
            self.acq_fn = acq_fn

        self.eval_acq = lambda params, model, likelihood: self.acq_fn(model, likelihood, params.reshape(-1, len(self.params)), self.params_samples, self.cost_samples, **acq_kwargs)


    def optimize(self,
                 trials: int = 10,
                 batch_size: int = 1,
                 num_iter_gp: int = 100,
                 plot: bool = False,
                 gpu: int = 0,
                 path: str = None,
                 n_restarts: int = 25,
                 lengthscale_prior: Dict[str, float] = dict(concentration=0.3, rate=1.),
                 lengthscale_constraint: float = 0.01,
                 mean_prior: Dict[str, float] = dict(loc=25., scale=2.),
                 noise_prior: Union[float, Dict[str, float]] = 1e-4) -> Tensor:
        '''Fn. for performing Bayesian optimization.

        Args:
            trials: number of trials (optimization budget)
            batch_size: batch_size of sampled params
            num_iter_gp: number of iterations to train GP surrogate model
            plot: plot results during training
            path: if not None where to save plots ans results
            n_restarts: restarts for scipy.optimize.minimize
            lengthscale_prior: dict. for alpha prior for lengthscale
                               of GP surrogate model
            lengthscale_constraint: minimal value for lengthscale
            mean_prior: Gaussian prior on mean of GP surrogate model
            noise_prior: either dict for alpha prior or float for constant noise

        Returns:
            Best parameters
        '''
        device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'
        round_tensor = lambda t, n_digits=3: torch.round(t * 10**n_digits) / (10**n_digits)
        results = {}

        model, likelihood = initialize_model(
            self.params_samples.to(device), self.cost_samples.flatten().to(device),
            num_iter_gp, lengthscale_prior, lengthscale_constraint, mean_prior, noise_prior
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
            # not as exact but needed for 2D

            next_params = self.propose_location(
                model, likelihood, self.eval_acq, self.params_space,
                self.bounds, n_restarts, batch_size
            )

            # check if next_params already exists
            mask = torch.tensor([(p != round_tensor(self.params_samples)).any() for p in round_tensor(next_params)])
            next_params = next_params[mask]

            # Obtain next noisy sample from the objective fct.
            # next_cost = self.obj_fn(self.dictionarize(next_params))
            next_cost = self.obj_fn(next_params)

            if plot:
                acquisition = self.eval_acq(self.params_space.numpy(), model, likelihood)
                _path = f"{path}/acq_plot_{i}.pdf" if path is not None else None
                if len(self.params) == 1:
                    plot_optimization(
                        model, likelihood, acquisition,
                        next_params, self.params_space.to(device),
                        self.params_samples, self.cost_samples, _path
                    )
                # elif len(self.params) == 2:
                #     self.plot_optimization_2d(path)
                else:
                    print('Plotting just up to 2 dimensions possible.')

            self.params_samples = torch.cat([self.params_samples, next_params], dim=0)
            self.cost_samples = torch.cat([self.cost_samples, next_cost], dim=0)

            model, likelihood = initialize_model(
                self.params_samples.to(device), self.cost_samples.flatten().to(device),
                num_iter_gp, lengthscale_prior, lengthscale_constraint,
                mean_prior, noise_prior
            )

            if path is not None:
                results[i] = {
                    "state_dict": model.state_dict(),
                    "p_samples": self.params_samples,
                    "c_samples": self.cost_samples
                }
                torch.save(results, f"{path}/results.pt")

        # if plot:
        #     _path = f"{path}/convergence_plot.pdf" if path is not None else None
        #     plot_convergence(self.params_samples, self.cost_samples, self.n_init, _path)

        return self.params_samples[np.argmin(self.cost_samples)]

    @staticmethod
    def propose_location(model: ExactGP,
                         likelihood: GaussianLikelihood,
                         eval_acq: Callable,
                         params_space: Tensor,
                         bounds: np.ndarray,
                         n_restarts: int = 25,
                         batch_size: int = 1):
        """Evaluate acquisition function to propose next params.

        Args:
            model: GP surrogate model
            likelihood: likeliood for GP
            eval_acq: evaluate acquisition function
                      eval_acq(model, likelihood, params)
            params_space: parameter space betwwen bounds of params
            bounds: bounds for parameter space
            n_restarts: restarts for scipy.optimize.minimize
            batch_size: batch_size of proposed parameters

        Returns:
            proposed next parameters
        """

        if params_space.size(-1) > 1:
            # multiple params, multiple samples
            next_params = propose_location_multidim(
                model, likelihood, eval_acq, bounds, n_restarts, batch_size
            )
        elif batch_size > 1:
            acquisition = eval_acq(params_space.numpy(), model, likelihood)
            # multiple samples, one param -> batched training of eval_fn
            # acq_peaks_idx = find_peaks_cwt(acquisition.flatten(), np.arange(1, 10))
            acq_peaks_idx = find_peaks(
                np.pad(acquisition.flatten(), (10,10), 'minimum'),
                prominence=0.1*(acquisition.max() - acquisition.min())
            )[0]
            acq_peaks_idx = np.clip(acq_peaks_idx - 10, 0, params_space.size(0) - 1)
            acq_peaks = acquisition[acq_peaks_idx.astype(np.int64)].flatten()
            acq_peaks_idx = acq_peaks_idx[acq_peaks.argsort()][-batch_size:]
            next_params = params_space[acq_peaks_idx]
        else:
            acquisition = eval_acq(params_space.numpy(), model, likelihood)
            # one sample, one param
            next_params = params_space[np.argmax(acquisition)].unsqueeze(0)

        return next_params

    # def dictionarize(self, params: np.ndarray) -> Dict[str, float]:
    #     return {name: float(val) for name, val in zip(self.params.keys(), params)}
