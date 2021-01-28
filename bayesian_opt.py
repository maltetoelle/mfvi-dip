import pickle

from ax.service.managed_loop import optimize
from ax.plot.contour import plot_contour
import fire

from super_resolution2 import super_resolution

RANDOM_SEED = 42

# IMG_NAME = "xray"
# NUM_ITER = 3000
# NET_SPECS = {"prior_mu": 0., "prior_sigma": 0.1, "kl_type": "forward"}
OPTIM_SPECS = {"lr": 0.01}


def bo(num_iter: int = 3000, total_trials: int = 20, img_name: str = "xray"):

    eval_fn = lambda params: super_resolution(beta=params["beta"],
                                              img_name=img_name,
                                              num_iter=num_iter,
                                              net_specs={
                                                "prior_mu": 0.,
                                                "prior_sigma": params["prior_sigma"],
                                                "kl_type": "forward"
                                              },
                                              optim_specs=OPTIM_SPECS,
                                              save=True)["psnr_gt"][-1]
    best_parameters, values, experiment, model = optimize(
        parameters=[
            {
                "name": "beta",
                "type": "range",
                "bounds": [1e-10, 1e-4]
            },
            {
                "name": "prior_sigma",
                "type": "range",
                "bounds": [0.02, 0.5]
            }
        ],
        evaluation_function=eval_fn,
        minimize=False,
        total_trials=total_trials,
        random_seed=RANDOM_SEED
    )

    plc = plot_contour(model=model, param_x='beta', param_y='prior_sigma', metric_name='objective')

    with open('bo_contour.obj', 'wb') as f:
        pickle.dump(plc, f)

    with open('best_params.txt', 'w') as f:
        f.write("beta: " + str(best_parameters["beta"]) + "\nprior_sigma: " + str(best_parameters["prior_sigma"]))

    print(best_parameters)

if __name__ == "__main__":
    fire.Fire(bo)
