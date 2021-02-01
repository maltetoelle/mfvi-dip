import os
import json
from datetime import datetime
from typing import Dict, Tuple, List
import time

import numpy as np
import fire
import torch
from torch import Tensor
from torch.nn import Module

from ax import (
    Parameter,
    RangeParameter,
    ParameterType,
    SearchSpace,
    SimpleExperiment,
    save
)
from ax.modelbridge.registry import Models


from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume

import warnings
warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

from super_resolution import super_resolution
from denoising import denoising
from inpainting import inpainting
from utils.bo_utils import generate_initial_data, initialize_model, optimize_qehvi_and_get_observation


class MultiObjDIPProb(Module):

    def __init__(self,
                 task: str,
                 config: Dict,
                 metrics: List[str] = ["uce", "lpips"],
                 img_name: str = "xray",
                 num_iter_eval_fn: int = 10000,
                 save_trials: bool = False,
                 path_log_dir: str = ".",
                 gpu: int = 0):
                 # params: List[str],
                 # metrics: List[str],
                 # bounds: List[float],
                 # ref_point: List[float],
                 # net_specs: Dict[str, Union[float, str]],
                 # optim_specs: Dict[str, float]):

        super(MultiObjDIPProb, self).__init__()

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
        self.metrics = metrics
        self.num_objectives = len(self.metrics)
        # self.params = config["net_params"] + config["optim_params"]
        self.params = [p["name"] for p in config["parameter"]]
        self.dim = len(self.params)

        # self.bounds = [(0.0, 1.0) for _ in range(len(params))]
        self.bounds = torch.tensor([p["bounds"] for p in config["parameter"]])
        self.ref_point = torch.tensor(config["ref_point"])

        self.img_name = img_name
        self.num_iter_eval_fn = num_iter_eval_fn
        self.save_trials = save_trials
        self.path_log_dir = path_log_dir
        self.gpu = gpu

    # make that able to run for batched data?
    def forward(self, param_vals: Tensor) -> Tensor:

        for i, param in enumerate(self.params):
            if param in self.optim_params:
                self.optim_specs[param] = param_vals[:,i]
            else:
                self.net_specs[param] = param_vals[:,i]

        # TODO: factor, sigma, kernel_type
        results = self.fn(img_name=self.img_name,
                          num_iter=self.num_iter_eval_fn,
                          net_specs=self.net_specs,
                          optim_specs=self.optim_specs,
                          save=self.save_trials,
                          path_log_dir=self.path_log_dir,
                          gpu=self.gpu)

        return torch.tensor([[results[metric] for metric in self.metrics]])


def bo(n_random: int = 5,
       trials: int = 15,
       minimize: bool = True,
       num_iter_eval_fn: int = 5000,
       metrics: List[str] = ["lpips", "uce"],
       img_name: str = "xray",
       task: str = 'denoising',
       config: str = "bo",
       save_trials: bool = False,
       log_dir: str = "/media/fastdata/toelle",
       gpu: int = 1,
       seed: int = 42,
       **kwargs):

    with open(config + ".json") as f:
        config = json.load(f)

    torch.manual_seed(seed)

    # logging
    # if task == "denoising":
    #     path_log_dir = "/bo_den/"
    # elif task == "super_resolution":
    #     path_log_dir = "/bo_sr/"
    # elif task == "inpainting":
    #     path_log_dir = "/bo_inp/"
    # if not os.path.exists(log_dir + path_log_dir):
    #     os.mkdir(log_dir + path_log_dir)
    #     print('here')
    # path_log_dir += "_".join(metrics) + "_" + datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    # if save_trials:
    #     path_log_dir = log_dir + path_log_dir
    #     os.mkdir(path_log_dir)
    #     print(path_log_dir)

    # could also pass one cfg let everything else be handled internally
    problem = MultiObjDIPProb(task=task,
                              config=config,
                              metrics=metrics,
                              img_name=img_name,
                              num_iter_eval_fn=num_iter_eval_fn,
                              save_trials=save_trials,
                              path_log_dir='.',
                              gpu=gpu)
                              # params: config["net_params"] + config["optim_params"],
                              # metrics: config["metrifc"],
                              # bounds: config["bounds"],
                              # net_specs: config["net_specs"],
                              # optim_specs: config["optim_specs"])

    N_TRIALS = 3
    N_BATCH = 1 # 25
    MC_SAMPLES = 128

    verbose = False

    hvs_qehvi_all = []

    hv = Hypervolume(ref_point=problem.ref_point)

    # average over multiple trials
    for trial in range(1, N_TRIALS + 1):
        torch.manual_seed(trial)

        print(f"\nTrial {trial:>2} of {N_TRIALS} ", end="")
        hvs_qehvi = []

        # call helper functions to generate initial training data and initialize model
        train_x_qehvi, train_obj_qehvi = generate_initial_data(problem, n=1)
        # mll_qparego, model_qparego = initialize_model(train_x_qparego, train_obj_qparego)

        # train_x_qehvi, train_obj_qehvi = train_x_qparego, train_obj_qparego
        # train_x_random, train_obj_random = train_x_qparego, train_obj_qparego
        # compute hypervolume
        mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi)

        # compute pareto front
        pareto_mask = is_non_dominated(train_obj_qehvi)
        pareto_y = train_obj_qehvi[pareto_mask]
        # compute hypervolume

        volume = hv.compute(pareto_y)

        hvs_qehvi.append(volume)

        # run N_BATCH rounds of BayesOpt after the initial random batch
        for iteration in range(1, N_BATCH + 1):

            t0 = time.time()

            # fit the models
            fit_gpytorch_model(mll_qehvi)

            # define the qEI and qNEI acquisition modules using a QMC sampler
            # qparego_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)
            qehvi_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)

            # optimize acquisition functions and get new observations
            new_x_qehvi, new_obj_qehvi = optimize_qehvi_and_get_observation(
                problem, model_qehvi, train_obj_qehvi, qehvi_sampler
            )

            # update training points
            train_x_qehvi = torch.cat([train_x_qehvi, new_x_qehvi])
            train_obj_qehvi = torch.cat([train_obj_qehvi, new_obj_qehvi])


            # update progress
            # compute pareto front
            pareto_mask = is_non_dominated(train_obj_qehvi)
            pareto_y = train_obj_qehvi[pareto_mask]
            # compute hypervolume
            volume = hv.compute(pareto_y)
            hvs_qehvi.append(volume)

            # reinitialize the models so they are ready for fitting on next iteration
            # Note: we find improved performance from not warm starting the model hyperparameters
            # using the hyperparameters from the previous iteration
            mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi)

            t1 = time.time()

            if verbose:
                print(
                    f"\nBatch {iteration:>2}: Hypervolume (random, qParEGO, qEHVI) = "
                    f"({hvs_random[-1]:>4.2f}, {hvs_qparego[-1]:>4.2f}, {hvs_qehvi[-1]:>4.2f}), "
                    f"time = {t1-t0:>4.2f}.", end=""
                )
            else:
                print(".", end="")

        hvs_qehvi_all.append(hvs_qehvi)


if __name__ == "__main__":
    fire.Fire(bo)
