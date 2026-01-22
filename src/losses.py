"""
Loss functions for E-I RNN fitting.

Implements:
- L_neuron: Trial-averaged activity (PSTH) matching
- L_trial: Single-trial variability matching (Sourmpis et al. 2023, 2026)
- L_reg: Weight regularization

Based on Sourmpis et al. (2023) "Trial matching" and Sourmpis et al. (2026).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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


def compute_L_trial(
    model_rates: torch.Tensor,
    target_rates: torch.Tensor,
    bin_size_ms: float = 25.0,
    smooth_ms: float = 32.0,
    mask: Optional[torch.Tensor] = None,
    recorded_indices: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute trial-matching loss.
    
    Matches single-trial population trajectories between model and data
    using greedy assignment (approximation to optimal transport).
    
    Args:
        model_rates: [batch, time, n_neurons] - RNN firing rates
        target_rates: [batch, time, n_recorded] - Recorded firing rates
        bin_size_ms: Bin size in milliseconds
        smooth_ms: Smoothing kernel size in milliseconds
        mask: [batch, time] - Valid timesteps (1) vs padding (0)
        recorded_indices: Which model neurons correspond to recorded neurons
    
    Returns:
        L_trial: Scalar loss
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
    
    # Greedy matching (differentiable approximation)
    # For each model trial, find closest target trial
    # This is a soft approximation using softmin
    
    # Temperature for soft assignment
    temperature = 0.1
    
    # Soft assignment weights
    soft_weights = F.softmax(-distances / temperature, dim=1)  # [batch, batch]
    
    # Weighted distance (soft matching)
    matched_distances = (soft_weights * distances).sum(dim=1)  # [batch]
    
    L_trial = matched_distances.mean()
    
    return L_trial


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
