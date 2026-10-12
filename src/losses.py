"""
Loss functions for E-I RNN fitting.

Implements:
- L_neuron: Trial-averaged activity (PSTH) matching
- L_trial: Single-trial variability matching (Sourmpis et al. 2023, 2026)
- L_reg: Weight regularization
- L_poisson: Poisson negative log-likelihood for spike count data
- Activity regularization

Based on Sourmpis et al. (2023) "Trial matching" and Sourmpis et al. (2026).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, List


def gaussian_kernel_1d(kernel_size: int, sigma: float) -> torch.Tensor:
    """Create 1D Gaussian kernel for temporal smoothing."""
    x = torch.arange(kernel_size).float() - kernel_size // 2
    kernel = torch.exp(-x**2 / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel


def smooth_temporal(x: torch.Tensor, kernel_size: int, dim: int = -2) -> torch.Tensor:
    """
    Apply temporal smoothing with Gaussian kernel.
    
    Args:
        x: Input tensor
        kernel_size: Size of smoothing kernel in bins
        dim: Dimension to smooth along (default: time dimension)
    
    Returns:
        Smoothed tensor
    """
    if kernel_size <= 1:
        return x
    
    # Create Gaussian kernel
    sigma = kernel_size / 4  # Standard choice
    kernel = gaussian_kernel_1d(kernel_size, sigma).to(x.device)
    
    # Reshape for conv1d: need [batch, channels, time]
    original_shape = x.shape
    
    # Move time dim to last, then reshape
    x = x.movedim(dim, -1)
    intermediate_shape = x.shape
    x = x.reshape(-1, 1, x.shape[-1])  # [batch*other, 1, time]
    
    # Pad and convolve
    padding = kernel_size // 2
    kernel = kernel.view(1, 1, -1)
    x_smooth = F.conv1d(x, kernel, padding=padding)
    
    # Reshape back
    x_smooth = x_smooth.reshape(intermediate_shape)
    x_smooth = x_smooth.movedim(-1, dim)
    
    return x_smooth


def compute_psth_correlation_loss(
    model_psth: torch.Tensor,
    target_psth: torch.Tensor,
    lambda_scale: float = 0.1,
    lambda_var: float = 0.05
) -> torch.Tensor:
    """
    Core PSTH correlation + scale + variance loss computation.

    This is extracted from compute_L_neuron for reuse in conditioned loss.

    Args:
        model_psth: [time, n_neurons] - Model PSTH (already trial-averaged)
        target_psth: [time, n_neurons] - Target PSTH (already trial-averaged)
        lambda_scale: Weight for scale matching loss
        lambda_var: Weight for variance matching loss

    Returns:
        loss: Scalar loss value
    """
    # === CORRELATION LOSS (shape matching) ===
    # Center the signals
    model_centered = model_psth - model_psth.mean(dim=0, keepdim=True)
    target_centered = target_psth - target_psth.mean(dim=0, keepdim=True)

    # Compute correlation per neuron
    numerator = (model_centered * target_centered).sum(dim=0)
    model_norm = (model_centered ** 2).sum(dim=0).sqrt().clamp(min=1e-6)
    target_norm = (target_centered ** 2).sum(dim=0).sqrt().clamp(min=1e-6)

    correlation = numerator / (model_norm * target_norm)

    # Loss: minimize (1 - correlation), averaged across neurons
    # Clamp correlation to [-1, 1] for numerical stability
    correlation = torch.clamp(correlation, -1.0, 1.0)
    L_corr = (1.0 - correlation).mean()

    # === SCALE LOSS (magnitude matching) ===
    # Penalize difference in mean firing rates
    model_mean = model_psth.mean()
    target_mean = target_psth.mean()
    target_var = target_psth.var().clamp(min=1e-6)

    L_scale = ((model_mean - target_mean) ** 2) / target_var

    # === VARIANCE LOSS (dynamics magnitude) ===
    # Encourage model to have similar temporal variance
    model_var = model_psth.var(dim=0).mean()
    target_var_per_neuron = target_psth.var(dim=0).mean()

    L_var = ((model_var - target_var_per_neuron) ** 2) / (target_var_per_neuron ** 2 + 1e-6)

    # Combine: prioritize correlation, but include scale and variance
    return L_corr + lambda_scale * L_scale + lambda_var * L_var


def compute_L_neuron(
    model_rates: torch.Tensor,
    target_rates: torch.Tensor,
    bin_size_ms: float = 25.0,
    smooth_ms: float = 8.0,
    mask: Optional[torch.Tensor] = None,
    recorded_indices: Optional[torch.Tensor] = None,
    lambda_scale: float = 0.1,
    lambda_var: float = 0.05
) -> torch.Tensor:
    """
    Compute neuron-wise PSTH loss using correlation + scale matching.

    This is more robust than z-score MSE when model variance is low.

    Args:
        model_rates: [batch, time, n_neurons] - RNN firing rates
        target_rates: [batch, time, n_recorded] - Recorded firing rates
        bin_size_ms: Bin size in milliseconds
        smooth_ms: Smoothing kernel size in milliseconds
        mask: [batch, time] - Valid timesteps (1) vs padding (0)
        recorded_indices: Which model neurons correspond to recorded neurons
        lambda_scale: Weight for scale matching loss (default 0.1)
        lambda_var: Weight for variance matching loss (default 0.05)

    Returns:
        L_neuron: Scalar loss
    """
    # Select recorded neurons from model if needed
    n_recorded = target_rates.shape[2]
    if recorded_indices is not None:
        model_rates = model_rates[:, :, recorded_indices]
    else:
        model_rates = model_rates[:, :, :n_recorded]

    # Apply mask
    if mask is not None:
        mask_expanded = mask.unsqueeze(-1)
        model_rates = model_rates * mask_expanded
        target_rates = target_rates * mask_expanded

    # Trial-average to get PSTH
    model_psth = model_rates.mean(dim=0)  # [time, neurons]
    target_psth = target_rates.mean(dim=0)

    # Temporal smoothing
    kernel_size = max(1, int(smooth_ms / bin_size_ms))
    model_psth = smooth_temporal(model_psth.unsqueeze(0), kernel_size, dim=1).squeeze(0)
    target_psth = smooth_temporal(target_psth.unsqueeze(0), kernel_size, dim=1).squeeze(0)

    # === CORRELATION LOSS (shape matching) ===
    # Center the signals
    model_centered = model_psth - model_psth.mean(dim=0, keepdim=True)
    target_centered = target_psth - target_psth.mean(dim=0, keepdim=True)

    # Compute correlation per neuron
    numerator = (model_centered * target_centered).sum(dim=0)
    model_norm = (model_centered ** 2).sum(dim=0).sqrt().clamp(min=1e-6)
    target_norm = (target_centered ** 2).sum(dim=0).sqrt().clamp(min=1e-6)

    correlation = numerator / (model_norm * target_norm)

    # Loss: minimize (1 - correlation), averaged across neurons
    # Clamp correlation to [-1, 1] for numerical stability
    correlation = torch.clamp(correlation, -1.0, 1.0)
    L_corr = (1.0 - correlation).mean()

    # === SCALE LOSS (magnitude matching) ===
    # Penalize difference in mean firing rates
    model_mean = model_psth.mean()
    target_mean = target_psth.mean()
    target_var = target_psth.var().clamp(min=1e-6)

    L_scale = ((model_mean - target_mean) ** 2) / target_var

    # === VARIANCE LOSS (dynamics magnitude) ===
    # Encourage model to have similar temporal variance
    model_var = model_psth.var(dim=0).mean()
    target_var_per_neuron = target_psth.var(dim=0).mean()

    L_var = ((model_var - target_var_per_neuron) ** 2) / (target_var_per_neuron ** 2 + 1e-6)

    # Combine: prioritize correlation, but include scale and variance
    L_neuron = L_corr + lambda_scale * L_scale + lambda_var * L_var

    return L_neuron


def compute_L_neuron_conditioned(
    model_rates: torch.Tensor,
    target_rates: torch.Tensor,
    trial_conditions: torch.Tensor,
    bin_size_ms: float = 25.0,
    smooth_ms: float = 8.0,
    mask: Optional[torch.Tensor] = None,
    recorded_indices: Optional[torch.Tensor] = None,
    lambda_scale: float = 0.1,
    lambda_var: float = 0.05,
    min_trials_per_condition: int = 3
) -> Tuple[torch.Tensor, dict]:
    """
    Compute PSTH loss separately for each experimental condition.

    Instead of computing one grand-average PSTH across all trials, this function
    computes separate PSTHs for each condition (e.g., location × reward × salience)
    and enforces matching for each. This preserves factor selectivity in the model.

    Args:
        model_rates: [n_trials, n_time, n_neurons] model firing rates
        target_rates: [n_trials, n_time, n_recorded] target firing rates
        trial_conditions: [n_trials] integer condition label for each trial (0 to n_conditions-1)
        bin_size_ms: Bin size in milliseconds
        smooth_ms: Smoothing kernel size in milliseconds
        mask: [n_trials, n_time] validity mask (optional)
        recorded_indices: Which model neurons correspond to recorded neurons
        lambda_scale: Weight for scale matching loss
        lambda_var: Weight for variance matching loss
        min_trials_per_condition: Minimum trials required per condition to compute loss

    Returns:
        loss: Mean loss across all valid conditions
        per_condition_loss: Dict of {condition: loss_value} for logging
    """
    # Select recorded neurons from model if needed
    n_recorded = target_rates.shape[2]
    if recorded_indices is not None:
        model_rates = model_rates[:, :, recorded_indices]
    else:
        model_rates = model_rates[:, :, :n_recorded]

    # Apply mask if provided
    if mask is not None:
        mask_expanded = mask.unsqueeze(-1)
        model_rates = model_rates * mask_expanded
        target_rates = target_rates * mask_expanded

    # Get unique conditions
    unique_conditions = torch.unique(trial_conditions)
    device = model_rates.device

    # Temporal smoothing kernel size
    kernel_size = max(1, int(smooth_ms / bin_size_ms))

    # Compute loss for each condition
    condition_losses = []
    per_condition_loss = {}

    for cond in unique_conditions:
        # Get trial indices for this condition
        cond_mask = trial_conditions == cond
        n_trials_cond = cond_mask.sum().item()

        # Skip conditions with too few trials
        if n_trials_cond < min_trials_per_condition:
            continue

        # Extract trials for this condition
        model_cond = model_rates[cond_mask]  # [n_trials_cond, time, neurons]
        target_cond = target_rates[cond_mask]

        # Compute condition-specific PSTH (trial average within condition)
        model_psth_cond = model_cond.mean(dim=0)  # [time, neurons]
        target_psth_cond = target_cond.mean(dim=0)

        # Apply temporal smoothing
        model_psth_cond = smooth_temporal(
            model_psth_cond.unsqueeze(0), kernel_size, dim=1
        ).squeeze(0)
        target_psth_cond = smooth_temporal(
            target_psth_cond.unsqueeze(0), kernel_size, dim=1
        ).squeeze(0)

        # Compute correlation-based loss for this condition
        loss_cond = compute_psth_correlation_loss(
            model_psth_cond, target_psth_cond,
            lambda_scale=lambda_scale, lambda_var=lambda_var
        )

        condition_losses.append(loss_cond)
        per_condition_loss[int(cond.item())] = loss_cond.item()

    # Average across conditions (equal weighting)
    if len(condition_losses) == 0:
        # Fallback to regular L_neuron if no valid conditions
        loss = compute_L_neuron(
            model_rates, target_rates, bin_size_ms, smooth_ms,
            mask=None,  # Already applied
            recorded_indices=None,  # Already selected
            lambda_scale=lambda_scale, lambda_var=lambda_var
        )
        per_condition_loss['fallback'] = loss.item()
    else:
        loss = torch.stack(condition_losses).mean()

    return loss, per_condition_loss


def compute_selectivity_index(
    rates: torch.Tensor,
    conditions: torch.Tensor,
    factor_values: torch.Tensor
) -> torch.Tensor:
    """
    Compute selectivity index (d-prime-like) for each neuron for a given factor.

    Args:
        rates: [n_trials, n_time, n_neurons] firing rates
        conditions: [n_trials] condition labels (not used directly, for compatibility)
        factor_values: [n_trials] binary factor values (0 or 1, e.g., low/high reward)

    Returns:
        selectivity: [n_neurons] selectivity index for each neuron
    """
    # Get trial-averaged rates for each neuron
    mean_rates = rates.mean(dim=1)  # [n_trials, n_neurons]

    # Split by factor level
    low_mask = factor_values == 0
    high_mask = factor_values == 1

    if low_mask.sum() < 3 or high_mask.sum() < 3:
        # Not enough trials for reliable estimate
        return torch.zeros(rates.shape[2], device=rates.device)

    low_rates = mean_rates[low_mask]  # [n_low, n_neurons]
    high_rates = mean_rates[high_mask]  # [n_high, n_neurons]

    # Compute d-prime: (mean_high - mean_low) / pooled_std
    mean_low = low_rates.mean(dim=0)
    mean_high = high_rates.mean(dim=0)
    var_low = low_rates.var(dim=0)
    var_high = high_rates.var(dim=0)

    # Pooled standard deviation
    n_low = low_rates.shape[0]
    n_high = high_rates.shape[0]
    pooled_var = ((n_low - 1) * var_low + (n_high - 1) * var_high) / (n_low + n_high - 2)
    pooled_std = torch.sqrt(pooled_var.clamp(min=1e-6))

    selectivity = (mean_high - mean_low) / pooled_std

    return selectivity


def compute_L_poisson(
    model_rates: torch.Tensor,
    target_rates: torch.Tensor,
    bin_size_ms: float = 25.0,
    smooth_ms: float = 8.0,
    mask: Optional[torch.Tensor] = None,
    recorded_indices: Optional[torch.Tensor] = None,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute Poisson negative log-likelihood loss.

    Spike counts follow Poisson statistics where variance equals mean.
    This loss properly handles the heteroscedasticity of neural data.

    L_poisson = mean(model_rates - target_rates * log(model_rates + eps))

    Args:
        model_rates: [batch, time, n_neurons] - RNN firing rates
        target_rates: [batch, time, n_recorded] - Recorded firing rates
        bin_size_ms: Bin size in milliseconds
        smooth_ms: Smoothing kernel size in milliseconds
        mask: [batch, time] - Valid timesteps (1) vs padding (0)
        recorded_indices: Which model neurons correspond to recorded neurons
        eps: Small constant for numerical stability

    Returns:
        L_poisson: Scalar loss
    """
    # Select recorded neurons from model if needed
    n_recorded = target_rates.shape[2]
    if recorded_indices is not None:
        model_rates = model_rates[:, :, recorded_indices]
    else:
        model_rates = model_rates[:, :, :n_recorded]

    # Apply mask
    if mask is not None:
        mask_expanded = mask.unsqueeze(-1)
        model_rates = model_rates * mask_expanded
        target_rates = target_rates * mask_expanded

    # Trial-average to get PSTH
    model_psth = model_rates.mean(dim=0)  # [time, neurons]
    target_psth = target_rates.mean(dim=0)

    # Temporal smoothing
    kernel_size = max(1, int(smooth_ms / bin_size_ms))
    model_psth = smooth_temporal(model_psth.unsqueeze(0), kernel_size, dim=1).squeeze(0)
    target_psth = smooth_temporal(target_psth.unsqueeze(0), kernel_size, dim=1).squeeze(0)

    # Ensure positive model rates
    model_psth = torch.clamp(model_psth, min=eps)

    # Poisson NLL: λ - k * log(λ) where λ = model rate, k = target rate
    L_poisson = (model_psth - target_psth * torch.log(model_psth + eps)).mean()

    return L_poisson


def compute_L_neuron_hybrid(
    model_rates: torch.Tensor,
    target_rates: torch.Tensor,
    bin_size_ms: float = 25.0,
    smooth_ms: float = 8.0,
    mask: Optional[torch.Tensor] = None,
    recorded_indices: Optional[torch.Tensor] = None,
    poisson_weight: float = 0.5,
    lambda_scale: float = 0.1,
    lambda_var: float = 0.05
) -> torch.Tensor:
    """
    Hybrid loss combining Poisson NLL (magnitude) with correlation (shape).

    L_hybrid = poisson_weight * L_poisson + (1 - poisson_weight) * L_corr

    This combines the statistical correctness of Poisson loss with the
    shape-matching properties of correlation loss.

    Args:
        model_rates: [batch, time, n_neurons] - RNN firing rates
        target_rates: [batch, time, n_recorded] - Recorded firing rates
        bin_size_ms: Bin size in milliseconds
        smooth_ms: Smoothing kernel size in milliseconds
        mask: [batch, time] - Valid timesteps (1) vs padding (0)
        recorded_indices: Which model neurons correspond to recorded neurons
        poisson_weight: Weight for Poisson loss (0-1)
        lambda_scale: Weight for scale loss in correlation component
        lambda_var: Weight for variance loss in correlation component

    Returns:
        L_hybrid: Scalar loss
    """
    L_poisson = compute_L_poisson(
        model_rates, target_rates, bin_size_ms, smooth_ms,
        mask, recorded_indices
    )

    L_corr_based = compute_L_neuron(
        model_rates, target_rates, bin_size_ms, smooth_ms,
        mask, recorded_indices, lambda_scale, lambda_var
    )

    L_hybrid = poisson_weight * L_poisson + (1 - poisson_weight) * L_corr_based

    return L_hybrid


def compute_activity_regularization(
    model_rates: torch.Tensor,
    target_mean: float = 10.0,
    target_max: float = 100.0,
    lambda_mean: float = 0.01,
    lambda_max: float = 0.001
) -> torch.Tensor:
    """
    Regularize network activity to biologically plausible range.

    Penalizes:
    - Deviation from target mean firing rate
    - Very high firing rates (above target_max)

    Args:
        model_rates: [batch, time, n_neurons] - RNN firing rates
        target_mean: Target mean firing rate in sp/s
        target_max: Maximum acceptable firing rate
        lambda_mean: Weight for mean rate penalty
        lambda_max: Weight for max rate penalty

    Returns:
        L_activity: Scalar regularization loss
    """
    # Mean rate penalty
    mean_rate = model_rates.mean()
    L_mean = lambda_mean * (mean_rate - target_mean) ** 2

    # Max rate penalty (soft hinge loss for rates above target_max)
    max_rates = model_rates.max(dim=1)[0].max(dim=0)[0]  # Max per neuron
    excess = torch.relu(max_rates - target_max)
    L_max = lambda_max * (excess ** 2).mean()

    return L_mean + L_max


def sinkhorn_assignment(
    distances: torch.Tensor,
    n_iters: int = 20,
    epsilon: float = 0.1
) -> torch.Tensor:
    """
    Sinkhorn-Knopp algorithm for differentiable optimal transport.
    Returns soft assignment matrix that approximates bijective matching.

    This enforces that each model trial matches to approximately one unique
    target trial (and vice versa), preventing the degenerate collapse where
    many model trials match to the same "easy" target.

    Args:
        distances: [n_trials, n_trials] pairwise distance matrix
        n_iters: number of Sinkhorn iterations (default 20)
        epsilon: entropy regularization (higher = softer assignment)

    Returns:
        P: [n_trials, n_trials] transport plan (doubly stochastic matrix)
    """
    # Convert distances to log-space cost matrix
    log_K = -distances / epsilon

    # Initialize dual variables
    log_u = torch.zeros(distances.shape[0], device=distances.device)
    log_v = torch.zeros(distances.shape[1], device=distances.device)

    # Sinkhorn iterations (in log-space for numerical stability)
    for _ in range(n_iters):
        log_u = -torch.logsumexp(log_K + log_v[None, :], dim=1)
        log_v = -torch.logsumexp(log_K + log_u[:, None], dim=0)

    # Compute transport plan
    log_P = log_K + log_u[:, None] + log_v[None, :]
    P = torch.exp(log_P)

    return P


def compute_L_trial(
    model_rates: torch.Tensor,
    target_rates: torch.Tensor,
    bin_size_ms: float = 25.0,
    smooth_ms: float = 32.0,
    mask: Optional[torch.Tensor] = None,
    recorded_indices: Optional[torch.Tensor] = None,
    sinkhorn_iters: int = 20,
    sinkhorn_epsilon: float = 0.1,
    use_poisson_distance: bool = False
) -> torch.Tensor:
    """
    Compute trial-matching loss using optimal transport (Sinkhorn algorithm).

    Matches single-trial population trajectories between model and data
    using the Sinkhorn algorithm for differentiable optimal transport.
    This enforces approximately bijective matching, preventing degenerate
    collapse where many model trials match to the same target.

    Args:
        model_rates: [batch, time, n_neurons] - RNN firing rates
        target_rates: [batch, time, n_recorded] - Recorded firing rates
        bin_size_ms: Bin size in milliseconds
        smooth_ms: Smoothing kernel size in milliseconds
        mask: [batch, time] - Valid timesteps (1) vs padding (0)
        recorded_indices: Which model neurons correspond to recorded neurons
        sinkhorn_iters: Number of Sinkhorn iterations (default 20)
        sinkhorn_epsilon: Entropy regularization for Sinkhorn (default 0.1)
        use_poisson_distance: If True, use Poisson divergence instead of Euclidean
            distance. This respects the natural variance structure of spike counts.

    Returns:
        L_trial: Scalar loss (optimal transport cost)
    """
    # Select recorded neurons from model if needed
    n_recorded = target_rates.shape[2]
    if recorded_indices is not None:
        model_rates = model_rates[:, :, recorded_indices]
    else:
        model_rates = model_rates[:, :, :n_recorded]

    # Population-average activity per trial
    model_pop = model_rates.mean(dim=2)   # [batch, time]
    target_pop = target_rates.mean(dim=2)

    # Apply mask if provided
    if mask is not None:
        model_pop = model_pop * mask
        target_pop = target_pop * mask

    # Temporal smoothing (coarser than L_neuron)
    kernel_size = max(1, int(smooth_ms / bin_size_ms))
    model_pop = smooth_temporal(model_pop, kernel_size, dim=1)
    target_pop = smooth_temporal(target_pop, kernel_size, dim=1)

    if use_poisson_distance:
        # Poisson Bregman divergence as distance metric
        # D(target || model) = target * log(target/model) - target + model
        # This is always non-negative and equals 0 when target = model
        eps = 1e-8
        model_expanded = model_pop.unsqueeze(1)  # [batch, 1, time]
        target_expanded = target_pop.unsqueeze(0)  # [1, batch, time]

        # Ensure positive rates
        model_expanded = torch.clamp(model_expanded, min=eps)
        target_expanded = torch.clamp(target_expanded, min=eps)

        # Bregman divergence for Poisson
        # D(t || m) = t * log(t/m) - t + m = t * log(t) - t * log(m) - t + m
        poisson_div = (
            target_expanded * torch.log(target_expanded + eps)
            - target_expanded * torch.log(model_expanded + eps)
            - target_expanded + model_expanded
        )
        distances = poisson_div.sum(dim=2)  # [batch, batch]

        # Normalize by number of time points for consistent scaling
        n_time = model_pop.shape[1]
        distances = distances / n_time
    else:
        # Original Euclidean distance on z-scored trajectories
        # Z-score normalize across trials (per timepoint)
        model_mean = model_pop.mean(dim=0, keepdim=True)
        model_std = model_pop.std(dim=0, keepdim=True) + 1e-8
        model_pop_norm = (model_pop - model_mean) / model_std

        target_mean = target_pop.mean(dim=0, keepdim=True)
        target_std = target_pop.std(dim=0, keepdim=True) + 1e-8
        target_pop_norm = (target_pop - target_mean) / target_std

        # Compute pairwise distances
        # distances[i,j] = ||model_trial_i - target_trial_j||
        distances = torch.cdist(model_pop_norm, target_pop_norm, p=2)  # [batch, batch]

    # Compute optimal transport plan using Sinkhorn algorithm
    # P is a doubly stochastic matrix (rows and columns sum to 1/n_trials)
    P = sinkhorn_assignment(distances, n_iters=sinkhorn_iters, epsilon=sinkhorn_epsilon)

    # Optimal transport cost: sum of (transport plan * distances)
    # Normalized by n_trials for consistent scaling
    n_trials = distances.shape[0]
    L_trial = (P * distances).sum() / n_trials

    return L_trial


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.01
):
    """
    Create a learning rate scheduler with linear warmup and cosine decay.

    Args:
        optimizer: PyTorch optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        min_lr_ratio: Minimum LR as fraction of initial LR

    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_L_reg(
    model: nn.Module,
    lambda_l2: float = 1e-4,
    lambda_sparse: float = 0.0
) -> torch.Tensor:
    """
    Compute weight regularization loss.
    
    Args:
        model: EIRNN model
        lambda_l2: L2 regularization strength
        lambda_sparse: L1/2 sparsity penalty (optional)
    
    Returns:
        L_reg: Scalar loss
    """
    W_rec = model.W_rec
    W_in = model.W_in
    
    # L2 regularization
    L_l2 = (W_rec ** 2).mean() + (W_in ** 2).mean()
    
    # Optional sparsity (L1/2 norm)
    if lambda_sparse > 0:
        L_sparse = (torch.abs(W_rec) ** 0.5).mean()
    else:
        L_sparse = torch.tensor(0.0, device=W_rec.device)
    
    L_reg = lambda_l2 * L_l2 + lambda_sparse * L_sparse
    
    return L_reg


def compute_L_output(
    model_outputs: torch.Tensor,
    target_outputs: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute output (behavioral) loss.
    
    Args:
        model_outputs: [batch, time, n_outputs] - Predicted outputs
        target_outputs: [batch, time, n_outputs] - Target outputs (e.g., eye position)
        mask: [batch, time] - Valid timesteps
    
    Returns:
        L_output: Scalar loss
    """
    if mask is not None:
        mask_expanded = mask.unsqueeze(-1)
        diff = (model_outputs - target_outputs) ** 2
        L_output = (diff * mask_expanded).sum() / mask_expanded.sum().clamp(min=1)
    else:
        L_output = ((model_outputs - target_outputs) ** 2).mean()
    
    return L_output


class EIRNNLoss(nn.Module):
    """
    Combined loss function for E-I RNN training.

    L = L_neuron + λ_trial * L_trial + λ_reg * L_reg [+ λ_output * L_output]

    Supports curriculum learning with adaptive weights.
    """

    def __init__(
        self,
        bin_size_ms: float = 25.0,
        lambda_reg: float = 1e-4,
        lambda_sparse: float = 0.0,
        lambda_output: float = 0.0,
        use_gradient_normalization: bool = True
    ):
        super().__init__()
        self.bin_size_ms = bin_size_ms
        self.lambda_reg = lambda_reg
        self.lambda_sparse = lambda_sparse
        self.lambda_output = lambda_output
        self.use_gradient_normalization = use_gradient_normalization

        # Curriculum learning parameters (can be modified during training)
        self.lambda_trial = 1.0  # Weight for trial-matching loss
        self.lambda_scale = 0.1  # Weight for scale loss in L_neuron
        self.lambda_var = 0.05   # Weight for variance loss in L_neuron

        # Running statistics for gradient normalization
        self.register_buffer('loss_ema', torch.ones(4))  # [L_neuron, L_trial, L_reg, L_output]
        self.ema_decay = 0.99
    
    def forward(
        self,
        model: nn.Module,
        model_rates: torch.Tensor,
        target_rates: torch.Tensor,
        model_outputs: Optional[torch.Tensor] = None,
        target_outputs: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        recorded_indices: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute total loss.
        
        Returns:
            loss: Total scalar loss
            components: Dict with individual loss components
        """
        # Compute individual losses
        L_neuron = compute_L_neuron(
            model_rates, target_rates,
            self.bin_size_ms, smooth_ms=8.0,
            mask=mask, recorded_indices=recorded_indices,
            lambda_scale=self.lambda_scale,
            lambda_var=self.lambda_var
        )

        L_trial = compute_L_trial(
            model_rates, target_rates,
            self.bin_size_ms, smooth_ms=32.0,
            mask=mask, recorded_indices=recorded_indices
        ) * self.lambda_trial  # Apply curriculum weight
        
        L_reg = compute_L_reg(model, self.lambda_reg, self.lambda_sparse)
        
        if self.lambda_output > 0 and model_outputs is not None and target_outputs is not None:
            L_output = compute_L_output(model_outputs, target_outputs, mask)
        else:
            L_output = torch.tensor(0.0, device=model_rates.device)
        
        # Store components (L_trial already has lambda_trial applied)
        components = {
            'L_neuron': L_neuron.item(),
            'L_trial': L_trial.item(),  # This is lambda_trial * raw_L_trial
            'L_reg': L_reg.item(),
            'L_output': L_output.item(),
            'lambda_trial': self.lambda_trial,
            'lambda_scale': self.lambda_scale,
            'lambda_var': self.lambda_var
        }
        
        # Combine losses
        if self.use_gradient_normalization:
            # Update EMA of loss magnitudes
            with torch.no_grad():
                current_losses = torch.tensor([
                    L_neuron.item(), L_trial.item(), 
                    L_reg.item(), L_output.item()
                ], device=self.loss_ema.device)
                self.loss_ema = self.ema_decay * self.loss_ema + (1 - self.ema_decay) * current_losses
            
            # Normalize by EMA (so gradients have similar magnitude)
            L_neuron_norm = L_neuron / (self.loss_ema[0] + 1e-8)
            L_trial_norm = L_trial / (self.loss_ema[1] + 1e-8)
            L_reg_norm = L_reg / (self.loss_ema[2] + 1e-8)
            L_output_norm = L_output / (self.loss_ema[3] + 1e-8)
            
            loss = L_neuron_norm + L_trial_norm + L_reg_norm + self.lambda_output * L_output_norm
        else:
            loss = L_neuron + L_trial + L_reg + self.lambda_output * L_output
        
        components['total'] = loss.item()
        
        return loss, components


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...")
    
    batch_size = 32
    n_steps = 100
    n_neurons = 50
    n_inputs = 14
    
    # Create dummy data
    model_rates = torch.rand(batch_size, n_steps, n_neurons) * 10
    target_rates = torch.rand(batch_size, n_steps, n_neurons) * 10
    
    # Test L_neuron
    L_neuron = compute_L_neuron(model_rates, target_rates)
    print(f"L_neuron: {L_neuron.item():.4f}")
    
    # Test L_trial
    L_trial = compute_L_trial(model_rates, target_rates)
    print(f"L_trial: {L_trial.item():.4f}")
    
    # Test with mask
    mask = torch.ones(batch_size, n_steps)
    mask[:, -20:] = 0  # Last 20 bins are padding
    
    L_neuron_masked = compute_L_neuron(model_rates, target_rates, mask=mask)
    print(f"L_neuron (masked): {L_neuron_masked.item():.4f}")
    
    # Test combined loss
    from model import EIRNN
    model = EIRNN(n_exc=40, n_inh=10, n_inputs=n_inputs)
    
    loss_fn = EIRNNLoss()
    inputs = torch.randn(batch_size, n_steps, n_inputs)
    model_rates, model_outputs = model(inputs)
    
    # Only fit to recorded neurons (first 50)
    target_rates_subset = target_rates[:, :, :50]
    loss, components = loss_fn(model, model_rates, target_rates_subset)
    
    print(f"\nCombined loss: {loss.item():.4f}")
    print("Components:", components)
    
    print("\nAll tests passed!")
