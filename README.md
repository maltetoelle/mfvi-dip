# Code for MIDL2021 Submission MFVI Deep Image Prior

Malte Tölle, Max-Heinrich Laves, Alexander Schlaefer

Code for our MIDL2021 submission *A Mean-Field Variational Inference Approach to Deep Image Prior for Inverse Problems in Medical Imaging*

## Abstract

Exploiting the deep image prior property of convolutional auto-encoder networks is especially interesting for medical image processing as it avoids hallucinations by omitting supervised learning. Its spectral bias towards lower frequencies makes it suitable for inverse image problems such as denoising and super-resolution, but manual early stopping has to be applied to act as a low-pass filter. In this paper, we present a novel Bayesian approach to deep image prior using mean-field variational inference. This allows for uncertainty quantification on a per-pixel level and, given the right prior distribution on the network weights, omits the need for early stopping. We optimize the parameters of the weight prior towards reconstruction accuracy using Bayesian optimization with Gaussian Process regression. We evaluate our approach on different inverse tasks on a variety of modalities and demonstrate that an optimized weight prior outperforms former state-of-the-art Bayesian deep image prior approaches. We show that a badly selected prior leads to worse accuracy and calibration and that it is sufficient to optimize the weight prior parameter per task domain.

## BibTeX

MIDL2021 (accepted)

```
@inproceedings{tolle2021mean,
  title={A Mean-Field Variational Inference Approach to Deep Image Prior for Inverse Problems in Medical Imaging},
  author={T{\"o}lle, Malte and Laves, Max-Heinrich and Schlaefer, Alexander},
  booktitle={Medical Imaging with Deep Learning},
  year={2021},
}
```

## How to run the code

```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

We provide bash scripts for all the experiments in the paper.
They can simply be run by e.g. `./train_inpainting.sh`, which trains all our different variants of the DIP (standard, with SGLD, MC dropout, and MFVI) for the specific task. You can specify two arguments if desired the first one being the path, where to save the results (default `experiments`), and second being the desired GPU to train on, which must be an integer (default `0`).
The experiments for Bayesian optimization of MFVI can be replicated by running e.g. `./bo_denoising.sh` with the same arguments as above. Because evaluation of the DIP can be distributed across multiple GPUs, the second argument must be a list e.g. `'[0,1]'` (default `'[0]'`).

All evaluations were performed using Jupyter notebooks, `eval.ipynb` for experiments and `eval_bo.ipynb` for evaluating the Bayesian optimization. You just have to adjust the paths where the experiments are stored and specify the task to evaluate.

## Contact

Malte Tölle  
[malte.toelle@gmail.com](mailto:malte.toelle@gmail.com)  

Max-Heinrich Laves  
[max-heinrich.laves@tuhh.de](mailto:max-heinrich.laves@tuhh.de)  
[@MaxLaves](https://twitter.com/MaxLaves)
