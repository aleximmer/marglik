from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.utils import parameters_to_vector
import logging

from laplace import KronLaplace
from laplace.curvature import AsdlGGN


def expand_prior_precision(prior_prec, model):
    theta = parameters_to_vector(model.parameters())
    device, P = theta.device, len(theta)
    assert prior_prec.ndim == 1
    if len(prior_prec) == 1:  # scalar
        return torch.ones(P, device=device) * prior_prec
    elif len(prior_prec) == P:  # full diagonal
        return prior_prec.to(device)
    else:
        return torch.cat([delta * torch.ones_like(m).flatten() for delta, m
                          in zip(prior_prec, model.parameters())])


def valid_performance(model, test_loader, likelihood, device):
    N = len(test_loader.dataset)
    perf = 0
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        if likelihood == 'classification':
            perf += (torch.argmax(model(X), dim=-1) == y).sum() / N
        else:
            perf += (model(X) - y).square().sum() / N
    return perf.item()


def marglik_optimization(model,
                         train_loader,
                         valid_loader=None,
                         likelihood='classification',
                         prior_structure='layerwise',
                         prior_prec_init=1.,
                         sigma_noise_init=1.,
                         temperature=1.,
                         n_epochs=500,
                         lr=1e-3,
                         lr_min=None,
                         optimizer='Adam',
                         scheduler='exp',
                         n_epochs_burnin=0,
                         n_hypersteps=100,
                         marglik_frequency=1,
                         lr_hyp=1e-1,
                         laplace=KronLaplace,
                         backend=AsdlGGN):
    """Runs marglik optimization training for a given model and training dataloader.

    Parameters
    ----------
    model : torch.nn.Module
        torch model
    train_loader : DataLoader
        pytorch training dataset loader
    valid_loader : DataLoader
    likelihood : str
        'classification' or 'regression'
    prior_structure : str
        'scalar', 'layerwise', 'diagonal'
    prior_prec_init : float
        initial prior precision
    sigma_noise_init : float
        initial observation noise (for regression only)
    temperature : float 
        factor for the likelihood for 'overcounting' data.
        Often required when using data augmentation.
    n_epochs : int
    lr : float
        learning rate for model optimizer
    lr_min : float
        minimum learning rate, defaults to lr and hence no decay
        to have the learning rate decay from 1e-3 to 1e-6, set
        lr=1e-3 and lr_min=1e-6.
    optimizer : str
        either 'Adam' or 'SGD'
    scheduler : str
        either 'exp' for exponential and 'cos' for cosine decay towards lr_min
    n_epochs_burnin : int default=0
        how many epochs to train without estimating and differentiating marglik
    n_hypersteps : int
        how many steps to take on the hyperparameters when marglik is estimated
    marglik_frequency : int
        how often to estimate (and differentiate) the marginal likelihood
    lr_hyp : float
        learning rate for hyperparameters (should be between 1e-3 and 1)
    laplace : Laplace
        type of Laplace approximation (Kron/Diag/Full)
    backend : Backend
        AsdlGGN/AsdlEF or BackPackGGN/BackPackEF

    Returns
    -------
    lap : Laplace
        lapalce approximation
    model : torch.nn.Module
    margliks : list
    losses : list
    """
    if lr_min is None:  # don't decay lr
        lr_min = lr
    device = parameters_to_vector(model.parameters()).device
    N = len(train_loader.dataset)
    H = len(list(model.parameters()))
    P = len(parameters_to_vector(model.parameters()))

    # differentiable hyperparameters
    hyperparameters = list()
    # prior precision
    log_prior_prec_init = np.log(temperature * prior_prec_init)
    if prior_structure == 'scalar':
        log_prior_prec = log_prior_prec_init * torch.ones(1, device=device)
    elif prior_structure == 'layerwise':
        log_prior_prec = log_prior_prec_init * torch.ones(H, device=device)
    elif prior_structure == 'diagonal':
        log_prior_prec = log_prior_prec_init * torch.ones(P, device=device)
    else:
        raise ValueError(f'Invalid prior structure {prior_structure}')
    log_prior_prec.requires_grad = True
    hyperparameters.append(log_prior_prec)

    # set up loss (and observation noise hyperparam)
    if likelihood == 'classification':
        criterion = CrossEntropyLoss(reduction='mean')
        sigma_noise = 1.
    elif likelihood == 'regression':
        criterion = MSELoss(reduction='mean')
        log_sigma_noise_init = np.log(sigma_noise_init)
        log_sigma_noise = log_sigma_noise_init * torch.ones(1, device=device)
        log_sigma_noise.requires_grad = True
        hyperparameters.append(log_sigma_noise)

    # set up model optimizer
    if optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr=lr)
    elif optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f'Invalid optimizer {optimizer}')

    # set up scheduler for lr decay
    n_steps = n_epochs * len(train_loader)
    if scheduler == 'exp':
        min_lr_factor = lr_min / lr
        gamma = np.exp(np.log(min_lr_factor) / n_steps)
        scheduler = ExponentialLR(optimizer, gamma=gamma)
    elif scheduler == 'cos':
        scheduler = CosineAnnealingLR(optimizer, n_steps, eta_min=lr_min)
    else:
        raise ValueError(f'Invalid scheduler {scheduler}')

    # set up hyperparameter optimizer
    hyper_optimizer = Adam(hyperparameters, lr=lr_hyp)

    best_marglik = np.inf
    best_model_dict = None
    best_precision = None
    losses = list()
    margliks = list()

    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0
        epoch_perf = 0
        
        # standard NN training per batch
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            if likelihood == 'regression':
                sigma_noise = torch.exp(log_sigma_noise).detach()
                crit_factor = temperature / (2 * sigma_noise.square())
            else:
                crit_factor = temperature
            prior_prec = torch.exp(log_prior_prec).detach()
            theta = parameters_to_vector(model.parameters())
            delta = expand_prior_precision(prior_prec, model)
            f = model(X)
            loss = criterion(f, y) + (0.5 * (delta * theta) @ theta) / N / crit_factor
            loss.backward()
            optimizer.step()
            epoch_loss += loss.cpu().item() / len(train_loader)
            if likelihood == 'regression':
                epoch_perf += (f.detach() - y).square().sum() / N
            else:
                epoch_perf += torch.sum(torch.argmax(f.detach(), dim=-1) == y).item() / N
            scheduler.step()
        losses.append(epoch_loss)

        # compute validation error to report during training
        if valid_loader is not None:
            with torch.no_grad():
                valid_perf = valid_performance(model, valid_loader, likelihood, device)
            logging.info(f'MARGLIK[epoch={epoch}]: network training. Loss={losses[-1]:.3f}; '
                         + f'Perf={epoch_perf:.3f}; Valid perf={valid_perf:.3f}; '
                         + f'lr={scheduler.get_last_lr()[0]:.7f}')
        else:
            logging.info(f'MARGLIK[epoch={epoch}]: network training. Loss={losses[-1]:.3f}; '
                         + f'Perf={epoch_perf:.3f}; lr={scheduler.get_last_lr()[0]:.7f}')

        # only update hyperparameters every "Frequency" steps after "burnin"
        if (epoch % marglik_frequency) != 0 or epoch < n_epochs_burnin:
            continue

        # optimizer hyperparameters by differentiating marglik
        # 1. fit laplace approximation
        sigma_noise = 1 if likelihood == 'classification' else torch.exp(log_sigma_noise)
        prior_prec = torch.exp(log_prior_prec)
        lap = laplace(model, likelihood, sigma_noise=sigma_noise, prior_precision=prior_prec,
                      temperature=temperature, backend=backend)
        lap.fit(train_loader)
        # 2. differentiate wrt. hyperparameters for n_hypersteps
        for _ in range(n_hypersteps):
            hyper_optimizer.zero_grad()
            if likelihood == 'classification':
                sigma_noise = None
            elif likelihood == 'regression':
                sigma_noise = torch.exp(log_sigma_noise)
            prior_prec = torch.exp(log_prior_prec)
            marglik = -lap.log_marginal_likelihood(prior_prec, sigma_noise)
            marglik.backward()
            hyper_optimizer.step()
            margliks.append(marglik.item())

        # early stopping on marginal likelihood
        if margliks[-1] < best_marglik:
            best_model_dict = deepcopy(model.state_dict())
            best_precision = deepcopy(prior_prec.detach())
            best_sigma = 1 if likelihood == 'classification' else deepcopy(sigma_noise.detach())
            best_marglik = margliks[-1]
            logging.info(f'MARGLIK[epoch={epoch}]: marglik optimization. MargLik={best_marglik:.2f}. '
                         + 'Saving new best model.')
        else:
            logging.info(f'MARGLIK[epoch={epoch}]: marglik optimization. MargLik={margliks[-1]:.2f}.'
                         + f'No improvement over {best_marglik:.2f}')

    logging.info('MARGLIK: finished training. Recover best model and fit Lapras.')
    if best_model_dict is not None:
        model.load_state_dict(best_model_dict)
        sigma_noise = best_sigma
        prior_prec = best_precision
    lap = laplace(model, likelihood, sigma_noise=sigma_noise, prior_precision=prior_prec,
                  temperature=temperature, backend=backend)
    lap.fit(train_loader)
    return lap, model, margliks, losses
