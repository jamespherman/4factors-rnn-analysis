"""
Extended loss functions for E-I RNN experiments.

Import these alongside the original losses:
    from src.losses import compute_L_neuron, compute_L_trial, compute_L_reg
    from src.losses_extended import compute_L_neuron_poisson, compute_L_activity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Import from original losses module
from src.losses import compute_L_neuron, smooth_temporal


def compute_L_neuron_poisson(
    model_rates: torch.Tensor,
    target_rates: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Poisson negative log-likelihood loss.
    
    Neural spike counts follow Poisson distribution; this is more appropriate
    than MSE which assumes Gaussian noise.
    
    Poisson NLL: model - target * log(model)
    """
    n_recorded = target_rates.shape[2]
    model_rates = model_rates[:, :, :n_recorded]
    
    model_rates_safe = torch.clamp(model_rates, min=eps)
    nll = model_rates_safe - target_rates * torch.log(model_rates_safe)
    
    if mask is not None:
        mask_expanded = mask.unsqueeze(-1)
        masked_nll = nll * mask_expanded
        return masked_nll.sum() / (mask.sum() * n_recorded + eps)
    else:
        return nll.mean()


def compute_L_neuron_hybrid(
    model_rates: torch.Tensor,
    target_rates: torch.Tensor,
    bin_size_ms: float = 25.0,
    smooth_ms: float = 8.0,
    mask: Optional[torch.Tensor] = None,
    poisson_weight: float = 0.5,
    correlation_weight: float = 0.5,
    eps: float = 1e-8
) -> torch.Tensor:
    """Hybrid loss combining Poisson NLL and correlation-based PSTH matching."""
    L_poisson = compute_L_neuron_poisson(model_rates, target_rates, mask, eps)
    L_corr = compute_L_neuron(
        model_rates, target_rates,
        bin_size_ms=bin_size_ms,
        smooth_ms=smooth_ms,
        mask=mask,
        normalize=True
    )
    return poisson_weight * L_poisson + correlation_weight * L_corr


def compute_L_activity(
    model_rates: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    target_mean: float = 10.0,
    target_max: float = 100.0,
    min_rate: float = 0.1
) -> torch.Tensor:
    """
    Activity regularization to keep firing rates in biological range.
    
    Penalizes:
    - Rates below min_rate (too quiet)
    - Rates above target_max (unrealistic)
    - Mean rate deviation from target_mean
    """
    if mask is not None:
        mask_expanded = mask.unsqueeze(-1)
        n_valid = mask.sum() * model_rates.shape[-1]
    else:
        n_valid = model_rates.numel()
    
    low_penalty = torch.relu(min_rate - model_rates).sum() / n_valid
    high_penalty = torch.relu(model_rates - target_max).sum() / n_valid
    
    if mask is not None:
        mean_rate = (model_rates * mask_expanded).sum() / n_valid
    else:
        mean_rate = model_rates.mean()
    mean_penalty = (mean_rate - target_mean).pow(2) / (target_mean ** 2)
    
    return low_penalty + high_penalty + 0.01 * mean_penalty


def compute_L_trial_sinkhorn(
    model_rates: torch.Tensor,
    target_rates: torch.Tensor,
    bin_size_ms: float = 25.0,
    smooth_ms: float = 32.0,
    mask: Optional[torch.Tensor] = None,
    sinkhorn_epsilon: float = 0.1,
    sinkhorn_iters: int = 20
) -> torch.Tensor:
    """Compute trial-matching loss with Sinkhorn optimal transport."""
    n_recorded = target_rates.shape[2]
    model_rates = model_rates[:, :, :n_recorded]
    
    model_pop = model_rates.mean(dim=2)
    target_pop = target_rates.mean(dim=2)
    
    if mask is not None:
        model_pop = model_pop * mask
        target_pop = target_pop * mask
    
    kernel_size = max(1, int(smooth_ms / bin_size_ms))
    model_pop = smooth_temporal(model_pop, kernel_size, dim=1)
    target_pop = smooth_temporal(target_pop, kernel_size, dim=1)
    
    model_mean = model_pop.mean(dim=0, keepdim=True)
    model_std = model_pop.std(dim=0, keepdim=True) + 1e-8
    model_pop_norm = (model_pop - model_mean) / model_std
    
    target_mean = target_pop.mean(dim=0, keepdim=True)
    target_std = target_pop.std(dim=0, keepdim=True) + 1e-8
    target_pop_norm = (target_pop - target_mean) / target_std
    
    C = torch.cdist(model_pop_norm, target_pop_norm, p=2)
    n = C.shape[0]
    
    K = torch.exp(-C / sinkhorn_epsilon)
    u = torch.ones(n, device=C.device) / n
    v = torch.ones(n, device=C.device) / n
    
    for _ in range(sinkhorn_iters):
        u = 1.0 / (K @ v + 1e-8)
        v = 1.0 / (K.T @ u + 1e-8)
    
    T = torch.diag(u) @ K @ torch.diag(v)
    L_trial = (T * C).sum()
    
    return L_trial
