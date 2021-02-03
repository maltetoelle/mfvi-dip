import os
import json
from datetime import datetime
from typing import Dict, Tuple, List

import numpy as np
import fire
import torch

from ax import (
    Parameter,
    RangeParameter,
    ParameterType,
    SearchSpace,
    SimpleExperiment,
    save
)
from ax.modelbridge.registry import Models
from ax.core.generator_run import GeneratorRun
from ax.core.arm import Arm

from super_resolution import super_resolution
from denoising import denoising
from inpainting import inpainting


def bo(trials: int = 20,
       minimize: bool = True,
       num_iter_eval_fn: int = 10000,
       metric: str = "psnr_gt",
       img_name: str = "xray",
       task: str = 'denoising',
       config: str = "bo_single",
       save_trials: bool = False,
       log_dir: str = "/media/fastdata/toelle",
       gpu: int = 1,
       seed: int = 42,
       **kwargs):

    with open(config + ".json") as f:
        config = json.load(f)

    torch.manual_seed(seed)

    if task == "denoising":
        fn = denoising
        path_log_dir = "/bo_den/"
    elif task == "super_resolution":
        fn = super_resolution
        path_log_dir = "/bo_sr/"
    elif task == "inpainting":
        fn = inpainting
        path_log_dir = "/bo_inp/"
    if not os.path.exists(log_dir + path_log_dir) and save_trials:
        os.mkdir(log_dir + path_log_dir)
    if not os.path.exists('./bo_exps'):
        os.mkdir('./bo_exps')

    timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    exp_path = "bo_exps/%s_%s.json" % (task, timestamp)
    path_log_dir += timestamp
    if save_trials:
        path_log_dir = log_dir + path_log_dir
        os.mkdir(path_log_dir)
        print(path_log_dir)
    # fn = denoising if task == 'denoising' else super_resolution

    NET_SPECS = config["net_specs"]
    OPTIM_SPECS = config["optim_specs"]

    def eval_fn(params: List[float]) -> Dict[str, Tuple[float]]:
        for p_name, p_val in params.items():
            if p_name in config["optim_params"]:
                OPTIM_SPECS[p_name] = p_val
            else:
                NET_SPECS[p_name] = p_val

        results = fn(img_name=img_name,
                     num_iter=num_iter_eval_fn,
                     net_specs=NET_SPECS,
                     optim_specs=OPTIM_SPECS,
                     save=save_trials,
                     path_log_dir=path_log_dir,
                     gpu=gpu,
                     **kwargs)

        # metric = config["metric"]
        # if metric[:4] == "psnr":
        #     psnr = results[metric]
        #     return {metric: (np.mean(psnr[-50:]), np.std(psnr[-100:]))}
        # else:
        res = results[metric][-int(0.1*num_iter_eval_fn):]
        return {metric: (-np.mean(res), np.std(res))}


    search_space = SearchSpace(
        parameters=[
        RangeParameter(name=p["name"], parameter_type=ParameterType.FLOAT, lower=p["bounds"][0], upper=p["bounds"][1]) for p in config["parameter"]
        ]
    )

    exp = SimpleExperiment(
        name=task,
        search_space=search_space,
        evaluation_function=eval_fn,
        objective_name=metric,
        minimize=minimize
    )

    # print(f"Starting random sampling with {n_random} samples...")
    print("Starting with initial params...")
    initial_params = []
    for intial_vals in config["intial_values"]:
        params = {p["name"]: val for p, val in zip(config["parameter"], intial_vals)}
        initial_params.append(params)

    initial_arms = [Arm(param) for param in initial_params]
    initial_generator_run = GeneratorRun(arms=initial_arms)

    # sobol = Models.SOBOL(exp.search_space)
    # exp.new_batch_trial(generator_run=sobol.gen(n_random))
    exp.new_batch_trial(generator_run=initial_generator_run)

    for i in range(trials):
        print(f"Starting trial {i}/{trials}...")
        intermediate_gp = Models.GPEI(experiment=exp, data=exp.eval())
        save(exp, exp_path)
        exp.new_trial(generator_run=intermediate_gp.gen(1))

    gp = Models.GPEI(experiment=exp, data=exp.eval())

    save(exp, exp_path)


if __name__ == "__main__":
    fire.Fire(bo)
