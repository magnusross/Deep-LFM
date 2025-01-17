# Deep Latent Force Models

This repository contains a PyTorch implementation of the *deep latent force model* (DLFM), presented in the paper, *Compositional Modeling of Nonlinear Dynamical Systems with ODE-based Random Features*. The DLFM takes the form of a deep Gaussian process with random feature expansions, but with the random Fourier features in question derived from a physics-informed ODE1 LFM kernel, rather than a more general choice (such as the exponentiated quadratic kernel).

![DLFM Model Architecture](assets/model.png "DLFM Model Architecture")

These compositions of physics-informed random features allow us to model nonlinearities in multivariate dynamical systems with a sound quantification of uncertainty and the ability to extrapolate effectively. The plot below shows DLFM predictions on a highly nonlinear multivariate time series, extracted from the [CHARIS PhysioNet dataset](https://physionet.org/content/charisdb/1.0.0/); note the ability of the model to extrapolate beyond the training regime which ends at t=0.7.

![PhysioNet Results](assets/physionet.png "PhysioNet Results")

## Usage

`requirements.txt` contains the small list of packages required to run `toy_demo.py`, which is identical to the toy data scenario described in our paper.

## Citation

```
@misc{mcdonald2021compositional,
      title={Compositional Modeling of Nonlinear Dynamical Systems with ODE-based Random Features}, 
      author={Thomas M. McDonald and Mauricio A. Álvarez},
      year={2021},
      eprint={2106.05960},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```
