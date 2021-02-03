import os
import sys
import json
from datetime import datetime
from typing import Dict, Tuple, List
from collections import OrderedDict
import time
import warnings
warnings.simplefilter("ignore")

import numpy as np
import fire
from tqdm import tqdm
import lpips
import torch
from torch import Tensor
from torch.nn import Module
import torch.multiprocessing as mp

# from ax import (
#     Parameter,
#     RangeParameter,
#     ParameterType,
#     SearchSpace,
#     SimpleExperiment,
#     save
# )
# from ax.modelbridge.registry import Models


from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
# from botorch.exceptions import BadInitialCandidatesWarning
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume

# import warnings
# warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
# warnings.filterwarnings('ignore', category=RuntimeWarning)

from super_resolution import super_resolution
from denoising import denoising
from inpainting import inpainting
from utils.bo_utils import generate_initial_data, initialize_model, optimize_and_get_observation
from train_utils import get_net_and_optim
from utils.bo_utils import ForkedPdb

num_input_channels = 32
num_output_channels = 2
num_channels_down = 128
num_channels_up = 128
num_channels_skip = 4
num_scales = 5
upsample_mode = 'bilinear'


class MultiObjDIPProb(Module):

    def __init__(self,
                 task: str,
                 config: Dict,
                 metrics: Dict[str, str] = OrderedDict({"uce": "min", "lpips": "min", "psnr_corrupted": "min"}),
                 img_name: str = "xray",
                 num_iter_eval_fn: int = 10000,
                 save_trials: bool = False,
                 path_log_dir: str = ".",
                 gpus: List[int] = [0, 1]):

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
        self._ops = {"max": 1, "min": int(-1)}
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
        self.gpus = gpus

        mp.set_start_method('spawn')

    def mp_fn(self, gpu, single_params, results, order, lpips_loss):
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

        # hard coded threshold for psnr noisy
        # must be at least this high to obtain a valid image
        if np.mean(single_result["psnr_corrupted"][-50:]) < 19:
            single_result["psnr_corrupted"][-50:] = [100] * 50

        results[order] = torch.tensor([[self._ops[op] * np.mean(single_result[metric][-50:]) for metric, op in self.metrics.items()]])
        # results[order] = torch.tensor([[np.mean(single_result[metric]), for metric in self.metrics]])

    # make that able to run for batched data?
    def forward(self, params: Tensor) -> Tensor:

        results = torch.empty(len(params), self.num_objectives)
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


def bo(n_trials: int = 3,
       n_batches: int = 25,
       n_initial: int = 5,
       mc_samples: int = 128,
       batch_size: int = 4,
       minimize: bool = True,
       num_iter_eval_fn: int = 3000,
       metrics: Dict[str, str] = OrderedDict({"uce": "min", "lpips": "min", "psnr_corrupted": "min"}),
       img_name: str = "xray",
       task: str = 'denoising',
       config: str = "bo",
       # save_trials: bool = False,
       # log_dir: str = "/media/fastdata/toelle",
       gpus: List[int] = [0,1],
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
    if not os.path.exists('./bo_exps'):
        os.mkdir('./bo_exps')
    fn_exp = f"{task}_{'_'.join(metrics)}_{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}"
    os.mkdir(f'./bo_exps/{fn_exp}')

    problem = MultiObjDIPProb(task=task,
                              config=config,
                              metrics=metrics,
                              img_name=img_name,
                              num_iter_eval_fn=num_iter_eval_fn,
                              save_trials=False,
                              path_log_dir='.',
                              gpus=gpus)


    hvs_all = []

    initial_train_x = torch.tensor([
        [1e-10, 0.4],
        [1e-6, 0.1],
        [1e-4, 0.05],
        [1e-8, 0.2]
    ])
    hv = Hypervolume(ref_point=problem.ref_point)

    # average over multiple trials
    for trial in range(1, n_trials + 1):
        torch.manual_seed(trial)

        print(f"\nTrial {trial:>2} of {n_trials} ")
        hvs = []

        # call helper functions to generate initial training data and initialize model
        train_x, train_obj = generate_initial_data(problem, train_x=initial_train_x, n=n_initial, seed=seed)

        # compute hypervolume
        mll, model = initialize_model(train_x, train_obj)

        # compute pareto front
        pareto_mask = is_non_dominated(train_obj)
        pareto_y = train_obj[pareto_mask]
        # compute hypervolume

        volume = hv.compute(pareto_y)

        hvs.append(volume)

        # pbar = tqdm(range(1, n_batches + 1))
        # run N_BATCH rounds of BayesOpt after the initial random batch
        for iteration in range(1, n_batches + 1):

            t0 = time.time()

            # fit the models
            fit_gpytorch_model(mll)

            # define the qEI and qNEI acquisition modules using a QMC sampler
            sampler = SobolQMCNormalSampler(num_samples=mc_samples)

            # optimize acquisition functions and get new observations
            new_x, new_obj = optimize_and_get_observation(
                problem, model, train_obj, sampler, batch_size
            )

            # update training points
            train_x = torch.cat([train_x, new_x])
            train_obj = torch.cat([train_obj, new_obj])


            # update progress
            # compute pareto front
            pareto_mask = is_non_dominated(train_obj)
            pareto_y = train_obj[pareto_mask]
            # compute hypervolume
            volume = hv.compute(pareto_y)
            hvs.append(volume)

            # reinitialize the models so they are ready for fitting on next iteration
            # Note: we find improved performance from not warm starting the model hyperparameters
            # using the hyperparameters from the previous iteration
            mll, model = initialize_model(train_x, train_obj)

            t1 = time.time()

            print(f"\nBatch {iteration:>2}: Hypervolume (, {hvs[-1]:>4.2f}), time = {t1-t0:>4.2f}.")

            torch.save({"train_x": train_x, "train_obj": train_obj, "hvs": hvs_all}, f'./bo_exps/{fn_exp}/train_data.pt')

        hvs_all.append(hvs)


if __name__ == "__main__":
    fire.Fire(bo)
