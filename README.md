# Deep Latent Force Models

This repository contains a PyTorch implementation of the *deep latent force model* (DLFM), presented in the paper, *Compositional Modeling of Nonlinear Dynamical Systems with ODE-based Random Features*. The DLFM takes the form of a deep Gaussian process with random feature expansions, but with the random Fourier features in question derived from a physics-informed ODE1 LFM kernel, rather than a more general choice (such as the exponentiated quadratic kernel).

![DLFM Model Architecture](assets/model.png "DLFM Model Architecture")

These compositions of physics-informed random features allow us to model nonlinearities in multivariate dynamical systems with a sound quantification of uncertainty and the ability to extrapolate effectively.

![PhysioNet Results](assets/physionet.png "PhysioNet Results")

## Usage

`requirements.txt` contains the small list of packages required to run `toy_demo.py`, which is identical to the toy data scenario described in our paper.

## Citation

TODO