# Marglik: Scalable Marginal Likelihood Estimation for Model Selection in Deep Learning

Code for the paper:

> Immer, Bauer, Fortuin, Rätsch, Khan. *Scalable Marginal Likelihood Estimation for Model Selection in Deep Learning.* ICML 2021.

The code is still work in progress, experiments will be added soon.

## Setup
Python 3.8 is required although it might work with later or earlier versions potentially.
Install minimal dependencies `pip install -r requirements.txt`.

## Usage
The training algorithm based on the marginal likelihood is implemented in `marglik.py`
and uses the [`laplace` package](https://github.com/AlexImmer/Laplace), a stand-alone package  which is based on my implementation for this paper
but provides additional features related to predictions and last-layer Laplace approximations.

