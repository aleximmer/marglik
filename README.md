# Marglik: Scalable Marginal Likelihood Estimation for Model Selection in Deep Learning

Code for the paper:

> Immer, Bauer, Fortuin, Rätsch, Khan. *Scalable Marginal Likelihood Estimation for Model Selection in Deep Learning.* ICML 2021.

In this repository, the Algorithm proposed in our paper is implemented in `marglik.py`.
The code uses the [`laplace-torch` package](https://github.com/AlexImmer/Laplace), a stand-alone package that originates from our implementation for this paper
but provides additional features related to predictions (see also [`linearized-predictives`](https://github.com/AlexImmer/BNN-predictions)) and last-layer Laplace approximations.

## Setup
Python 3.8 is required although it might work with later or earlier versions potentially.
Install minimal dependencies `pip install -r requirements.txt`.

## Usage

The method `marglik_optimization(...)` in `marglik.py` requires at least a pytorch model and training loader.
The method trains the neural network and optimizes the prior precision and observation noise (for regression).
For details, see the signature of the method or consult the documentation of the separately published `laplace-torch` library [here](https://aleximmer.github.io/Laplace/).
