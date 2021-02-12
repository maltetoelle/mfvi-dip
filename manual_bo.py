import os
from typing import Dict, List, Tuple
import json
from datetime import datetime
import warnings
warnings.simplefilter("ignore")
import numpy as np
import fire

import torch
import torch.nn as nn
import torch.multiprocessing as mp

import sklearn.gaussian_process as gp

from bayesian_optimization import BayesianOptimization

from utils.bo_utils import BatchedDIPProblem
from super_resolution import super_resolution
from denoising import denoising
from inpainting import inpainting


def bo(
    exp_name: str = "bo",
    trials: int = 10,
    num_iter_eval_fn: int = 10000,
    batch_size: int = None,
    num_iter_gp: int = 1000,
    n_init: int = 4,
    criterion: str = 'nll',
    metric: str = "psnr_gt_sm",
    img_name: str = "xray",
    task: str = 'denoising',
    config: str = "./configs/bo_prior_sigma",
    log_dir: str = "../bo_exps",
    trials_log_dir: str = None,
    gpus: List[int] = [0, 1],
    seed: int = 11):

    torch.manual_seed(seed)
    np.random.seed(seed)

    with open(config + ".json") as f:
        config = json.load(f)

    if log_dir is not None:
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        log_dir = f"{log_dir}/{exp_name}"
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

    # TODO: make that also False when None for path_log_dir in fns
    save_trials = False if trials_log_dir is None else True
    batch_size = len(gpus) if batch_size is None else batch_size

    # def eval_fn(params: Dict[str, float]) -> List[float]:
    #     for p_name, p_val in params.items():
    #         if p_name in config["optim_params"]:
    #             OPTIM_SPECS[p_name] = p_val
    #         else:
    #             NET_SPECS[p_name] = p_val
    #
    #     results = fn(img_name=img_name,
    #                  num_iter=num_iter_eval_fn,
    #                  criterion=criterion,
    #                  net_specs=NET_SPECS,
    #                  optim_specs=OPTIM_SPECS,
    #                  save=save_trials,
    #                  path_log_dir=trials_log_dir,
    #                  gpu=gpu)
    #
    #     res = results[metric][-int(0.1*num_iter_eval_fn):]
    #     return [np.mean(res)]


    params = {p["name"]: p["bounds"] for p in config["parameter"]}
    lengthscale_prior = config["lengthscale_prior"] if "lengthscale_prior" in list(config.keys()) else dict(concentration=0.3, rate=1.)
    lengthscale_constraint = config["lengthscale_constraint"] if "lengthscale_constraint" in list(config.keys()) else 0.05
    #  25. only for denoising the rest is lower
    mean_prior = config["mean_prior"] if "mean_prior" in list(config.keys()) else dict(loc=25., scale=2.)
    noise_prior = config["noise_prior"] if "noise_prior" in list(config.keys()) else dict(concentration=1e-2, rate=100.)


    initial_params_vals = config["initial_parameter"] if "initial_parameter" in config.keys() else None

    acq_kwargs = {"xi": 0.1}

    batched_dip_prob = BatchedDIPProblem(
        path_log_dir=trials_log_dir, gpus=gpus, task=task, config=config,
        num_iter_eval_fn=num_iter_eval_fn, save_trials=save_trials,
        img_name=img_name, metric=metric, seed=None
    )

    bayesian_optimization = BayesianOptimization(
        params=params,
        initial_params_vals=initial_params_vals,
        n_init=n_init,
        obj_fn=batched_dip_prob, # eval_fn,
        acq_fn='expected_improvement',
        acq_kwargs=acq_kwargs
    )

    best_params = bayesian_optimization.optimize(
        trials=trials, plot=True, gpu=gpus[0], path=log_dir,
        lengthscale_prior=lengthscale_prior, mean_prior=mean_prior,
        noise_prior=noise_prior, lengthscale_constraint=lengthscale_constraint,
        num_iter_gp=num_iter_gp, batch_size=batch_size
    )

    print(best_params)


if __name__ == "__main__":
    fire.Fire(bo)
