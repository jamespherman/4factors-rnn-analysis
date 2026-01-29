"""
Phase 5 Experiment Script - Limitation Analysis

Experiments:
0. Extended training (1000 epochs, patience=150) - run in background
1. Faster dynamics (τ=25ms)
2. Higher noise (scale=0.3 and 0.5)
3. Relaxed E/I assignment (bypass_dale=True)
4. Learnable τ (revisited)
5. Poisson L_trial (revisited)
6. Low-rank connectivity (rank=50)

Plus analytical deep-dives on I neurons and spectral properties.

Usage:
    python scripts/experiment_phase5.py --data data/rnn_export_Newton_08_15_2025_SC.mat --output results/phase5/
    python scripts/experiment_phase5.py --data data/rnn_export_Newton_08_15_2025_SC.mat --output results/phase5/ --experiments extended_training
"""

import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import pearsonr

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import EIRNN, create_model_from_data
from src.losses import (
    compute_L_neuron, compute_L_trial, compute_L_reg,
    get_cosine_schedule_with_warmup
)
from src.data_loader import load_session, train_val_split


def compute_psth_correlation(
    model: EIRNN,
    data: dict,
    device: str
) -> float:
    """Compute mean PSTH correlation across neurons."""
    model.eval()

    with torch.no_grad():
        inputs = data['inputs'].to(device)
        targets = data['targets'].to(device)

        model_rates, _ = model(inputs)

        # Trial-average
        model_psth = model_rates.mean(dim=0).cpu().numpy()
        target_psth = targets.mean(dim=0).cpu().numpy()

        # Correlation per neuron
        n_recorded = target_psth.shape[1]
        correlations = []
        for i in range(n_recorded):
            r = np.corrcoef(model_psth[:, i], target_psth[:, i])[0, 1]
            if not np.isnan(r):
                correlations.append(r)

        return np.mean(correlations) if correlations else 0.0


def compute_per_neuron_correlations(
    model: EIRNN,
    data: dict,
    device: str
) -> np.ndarray:
    """Compute PSTH correlation for each neuron."""
    model.eval()

    with torch.no_grad():
        inputs = data['inputs'].to(device)
        targets = data['targets'].to(device)

        model_rates, _ = model(inputs)

        # Trial-average
        model_psth = model_rates.mean(dim=0).cpu().numpy()
        target_psth = targets.mean(dim=0).cpu().numpy()

        # Correlation per neuron
        n_recorded = target_psth.shape[1]
        correlations = np.zeros(n_recorded)
        for i in range(n_recorded):
            r = np.corrcoef(model_psth[:, i], target_psth[:, i])[0, 1]
            correlations[i] = r if not np.isnan(r) else 0.0

        return correlations


def compute_fano_factors(
    model: EIRNN,
    data: dict,
    device: str
) -> tuple:
    """Compute Fano factors for model and real data."""
    model.eval()

    with torch.no_grad():
        inputs = data['inputs'].to(device)
        targets = data['targets'].to(device)

        model_rates, _ = model(inputs)

        # Compute Fano factor (variance/mean) across trials
        model_var = model_rates.var(dim=0).cpu().numpy()
        model_mean = model_rates.mean(dim=0).cpu().numpy()
        target_var = targets.var(dim=0).cpu().numpy()
        target_mean = targets.mean(dim=0).cpu().numpy()

        # Avoid division by zero
        model_fano = np.where(model_mean > 0.1, model_var / model_mean, 0)
        target_fano = np.where(target_mean > 0.1, target_var / target_mean, 0)

        return model_fano.mean(), target_fano.mean()


def train_with_config(
    config: dict,
    train_data: dict,
    val_data: dict,
    neuron_info: dict,
    n_inputs: int,
    bin_size_ms: float,
    device: str = 'cpu',
    seed: int = 42
) -> dict:
    """Train a model with the given configuration."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create model
    model = create_model_from_data(
        n_classic=neuron_info['n_exc'],
        n_interneuron=neuron_info['n_inh'],
        n_inputs=n_inputs,
        enforce_ratio=True,
        dt=bin_size_ms,
        tau=config.get('tau', 50.0),
        noise_scale=config.get('noise_scale', 0.1),
        learnable_tau=config.get('learnable_tau', 'none'),
        tau_e_init=config.get('tau_e_init', 50.0),
        tau_i_init=config.get('tau_i_init', 35.0),
        learnable_alpha=config.get('learnable_alpha', 'none'),
        alpha_init=config.get('alpha_init', 0.5),
        alpha_e_init=config.get('alpha_e_init', None),
        alpha_i_init=config.get('alpha_i_init', None),
        input_embed_dim=config.get('input_embed_dim', None),
        input_embed_type=config.get('input_embed_type', 'learnable'),
        attention_heads=config.get('attention_heads', 4),
        learnable_h0=config.get('learnable_h0', False),
        h0_init=config.get('h0_init', 0.1),
        low_rank=config.get('low_rank', None),
        bypass_dale=config.get('bypass_dale', False),
        device=device
    )

    # Print model info
    name = config.get('name', 'unknown')
    print(f"  [{name}] Config: tau={config.get('tau', 50)}, noise={config.get('noise_scale', 0.1)}")
    if config.get('learnable_h0', False):
        print(f"  [{name}] Learnable initial state: h0_init={config.get('h0_init', 0.1)}")
    if config.get('learnable_tau', 'none') != 'none':
        print(f"  [{name}] Learnable tau: mode={config['learnable_tau']}")
    if config.get('bypass_dale', False):
        print(f"  [{name}] *** DALE'S LAW BYPASSED ***")
    if config.get('low_rank') is not None:
        print(f"  [{name}] Low-rank constraint: rank={config['low_rank']}")

    # Optimizer
    lr = config.get('lr', 1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-5
    )

    # Loss configuration
    use_grad_balancing = config.get('use_grad_balancing', True)
    ltrial_scale = config.get('ltrial_scale', 0.5)
    lambda_reg = config.get('lambda_reg', 1e-4)
    use_poisson_ltrial = config.get('use_poisson_ltrial', False)

    # For gradient balancing, track EMAs
    loss_ema = {'L_neuron': 1.0, 'L_trial': 1.0}
    ema_decay = 0.99

    # Training loop
    max_epochs = config.get('max_epochs', 500)
    patience = config.get('patience', 100)
    best_val_corr = float('-inf')
    epochs_without_improvement = 0
    best_model_state = None

    history = {
        'train_corr': [],
        'val_corr': [],
        'train_loss': [],
        'val_loss': [],
        'lr': [],
        'tau_values': [],
        'h0_values': [],
        'fano_model': [],
        'fano_real': []
    }

    pbar = tqdm(range(max_epochs), desc=config.get('name', 'Training'))
    for epoch in pbar:
        # Train
        model.train()
        inputs = train_data['inputs'].to(device)
        targets = train_data['targets'].to(device)
        mask = train_data['mask'].to(device)

        optimizer.zero_grad()
        model_rates, _ = model(inputs)
        n_recorded = targets.shape[2]
        model_rates_recorded = model_rates[:, :, :n_recorded]

        # Compute losses
        L_neuron = compute_L_neuron(
            model_rates_recorded, targets, bin_size_ms,
            mask=mask, lambda_scale=0.1, lambda_var=0.05
        )

        L_trial_raw = compute_L_trial(
            model_rates_recorded, targets, bin_size_ms,
            mask=mask, sinkhorn_iters=20, sinkhorn_epsilon=0.1,
            use_poisson_distance=use_poisson_ltrial
        )

        L_reg = compute_L_reg(model, lambda_reg)

        # Gradient balancing
        if use_grad_balancing:
            with torch.no_grad():
                loss_ema['L_neuron'] = ema_decay * loss_ema['L_neuron'] + (1 - ema_decay) * L_neuron.item()
                loss_ema['L_trial'] = ema_decay * loss_ema['L_trial'] + (1 - ema_decay) * L_trial_raw.item()

            L_neuron_norm = L_neuron / (loss_ema['L_neuron'] + 1e-8)
            L_trial_norm = L_trial_raw / (loss_ema['L_trial'] + 1e-8)
            loss = L_neuron_norm + ltrial_scale * L_trial_norm + L_reg
        else:
            loss = L_neuron + 0.5 * L_trial_raw + L_reg

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss = loss.item()
        train_corr = compute_psth_correlation(model, train_data, device)

        # Validate
        model.eval()
        with torch.no_grad():
            inputs_val = val_data['inputs'].to(device)
            targets_val = val_data['targets'].to(device)
            mask_val = val_data['mask'].to(device)

            model_rates_val, _ = model(inputs_val)
            model_rates_val_recorded = model_rates_val[:, :, :n_recorded]

            L_neuron_val = compute_L_neuron(
                model_rates_val_recorded, targets_val, bin_size_ms,
                mask=mask_val, lambda_scale=0.1, lambda_var=0.05
            )

            val_loss_value = L_neuron_val.item()
            val_corr = compute_psth_correlation(model, val_data, device)

        # Update scheduler
        scheduler.step(val_loss_value)

        # Record history
        history['train_corr'].append(train_corr)
        history['val_corr'].append(val_corr)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss_value)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Record tau values if learnable
        if hasattr(model, 'get_tau_values'):
            history['tau_values'].append(model.get_tau_values())

        # Record h0 values if learnable
        if model.h0 is not None:
            h0_vals = model.h0.detach().cpu().numpy()
            history['h0_values'].append({
                'mean': float(h0_vals.mean()),
                'std': float(h0_vals.std()),
                'min': float(h0_vals.min()),
                'max': float(h0_vals.max())
            })

        # Record Fano factors periodically
        if epoch % 50 == 0:
            fano_model, fano_real = compute_fano_factors(model, val_data, device)
            history['fano_model'].append(fano_model)
            history['fano_real'].append(fano_real)

        # Update progress
        pbar.set_postfix({
            'train_corr': f'{train_corr:.3f}',
            'val_corr': f'{val_corr:.3f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.1e}'
        })

        # Early stopping
        if val_corr > best_val_corr:
            best_val_corr = val_corr
            epochs_without_improvement = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Get final values
    tau_values = model.get_tau_values() if hasattr(model, 'get_tau_values') else {}
    alpha_values = model.get_alpha_values() if hasattr(model, 'get_alpha_values') else {}
    h0_values = {}
    if model.h0 is not None:
        h0_vals = model.h0.detach().cpu().numpy()
        h0_values = {
            'mean': float(h0_vals.mean()),
            'std': float(h0_vals.std()),
            'min': float(h0_vals.min()),
            'max': float(h0_vals.max())
        }

    return {
        'best_val_corr': best_val_corr,
        'final_val_corr': val_corr,
        'history': history,
        'epochs_trained': epoch + 1,
        'tau_values': tau_values,
        'alpha_values': alpha_values,
        'h0_values': h0_values,
        'config': config,
        'best_model_state': best_model_state
    }


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_i_neurons(
    model: EIRNN,
    data: dict,
    neuron_info: dict,
    device: str,
    output_dir: Path
):
    """
    Deep analysis of I neuron fitting.

    Computes:
    - Input-output correlations
    - Temporal autocorrelation
    - Signal-to-noise analysis
    """
    model.eval()
    analysis_dir = output_dir / 'analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        inputs = data['inputs'].to(device)
        targets = data['targets'].to(device)

        model_rates, _ = model(inputs)

        # Get PSTHs
        model_psth = model_rates.mean(dim=0).cpu().numpy()
        target_psth = targets.mean(dim=0).cpu().numpy()
        inputs_np = inputs.cpu().numpy()

    n_recorded = target_psth.shape[1]
    n_inputs = inputs_np.shape[2]
    n_exc = neuron_info['n_exc']
    n_inh = neuron_info['n_inh']

    # Indices for E and I neurons
    e_indices = np.arange(n_exc)
    i_indices = np.arange(n_exc, n_exc + n_inh)

    # Ensure we only use indices that exist in recorded neurons
    e_indices = e_indices[e_indices < n_recorded]
    i_indices = i_indices[i_indices < n_recorded]

    # 1. Input-output correlations for I neurons
    print("Computing input-output correlations for I neurons...")
    input_names = [
        'fixation', 'loc1', 'loc2', 'loc3', 'loc4',
        'go', 'reward', 'eye_x', 'eye_y',
        'face', 'nonface', 'bullseye', 'high_sal', 'low_sal'
    ]

    # Trial-averaged inputs
    input_psth = inputs_np.mean(axis=0)  # [time, n_inputs]

    correlations_i = np.zeros((len(i_indices), n_inputs))
    correlations_e = np.zeros((len(e_indices), n_inputs))

    for i, idx in enumerate(i_indices):
        for j in range(n_inputs):
            r, _ = pearsonr(target_psth[:, idx], input_psth[:, j])
            correlations_i[i, j] = r if not np.isnan(r) else 0

    for i, idx in enumerate(e_indices):
        for j in range(n_inputs):
            r, _ = pearsonr(target_psth[:, idx], input_psth[:, j])
            correlations_e[i, j] = r if not np.isnan(r) else 0

    # Plot input correlations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # I neuron correlations
    ax = axes[0]
    im = ax.imshow(correlations_i.T, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    ax.set_yticks(range(n_inputs))
    ax.set_yticklabels(input_names)
    ax.set_xlabel('I Neuron Index')
    ax.set_title(f'I Neuron Input Correlations (n={len(i_indices)})')
    plt.colorbar(im, ax=ax, label='Correlation')

    # Mean correlations comparison
    ax = axes[1]
    mean_corr_i = np.abs(correlations_i).mean(axis=0)
    mean_corr_e = np.abs(correlations_e).mean(axis=0)
    x = np.arange(n_inputs)
    width = 0.35
    ax.bar(x - width/2, mean_corr_e, width, label='E neurons', color='blue', alpha=0.7)
    ax.bar(x + width/2, mean_corr_i, width, label='I neurons', color='red', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(input_names, rotation=45, ha='right')
    ax.set_ylabel('Mean |Correlation|')
    ax.set_title('Input Responsiveness by Neuron Type')
    ax.legend()

    plt.tight_layout()
    plt.savefig(analysis_dir / 'i_neuron_input_correlations.png', dpi=150)
    plt.close()

    # 2. Autocorrelation analysis
    print("Computing temporal autocorrelation...")
    max_lag = 20  # 20 bins = 500ms

    def compute_autocorr(x, max_lag):
        """Compute normalized autocorrelation."""
        x = x - x.mean()
        result = np.correlate(x, x, mode='full')
        result = result[len(result)//2:]
        result = result / result[0]
        return result[:max_lag+1]

    autocorr_real_e = []
    autocorr_real_i = []
    autocorr_model_e = []
    autocorr_model_i = []

    for idx in e_indices:
        autocorr_real_e.append(compute_autocorr(target_psth[:, idx], max_lag))
        autocorr_model_e.append(compute_autocorr(model_psth[:, idx], max_lag))

    for idx in i_indices:
        autocorr_real_i.append(compute_autocorr(target_psth[:, idx], max_lag))
        autocorr_model_i.append(compute_autocorr(model_psth[:, idx], max_lag))

    # Plot autocorrelation
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    lags = np.arange(max_lag + 1) * 25  # Convert to ms

    ax = axes[0]
    if autocorr_real_e:
        ax.plot(lags, np.mean(autocorr_real_e, axis=0), 'b-', label='E real', linewidth=2)
        ax.fill_between(lags,
                        np.mean(autocorr_real_e, axis=0) - np.std(autocorr_real_e, axis=0),
                        np.mean(autocorr_real_e, axis=0) + np.std(autocorr_real_e, axis=0),
                        alpha=0.2, color='blue')
    if autocorr_real_i:
        ax.plot(lags, np.mean(autocorr_real_i, axis=0), 'r-', label='I real', linewidth=2)
        ax.fill_between(lags,
                        np.mean(autocorr_real_i, axis=0) - np.std(autocorr_real_i, axis=0),
                        np.mean(autocorr_real_i, axis=0) + np.std(autocorr_real_i, axis=0),
                        alpha=0.2, color='red')
    ax.set_xlabel('Lag (ms)')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Real Data: E vs I Autocorrelation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if autocorr_model_e:
        ax.plot(lags, np.mean(autocorr_model_e, axis=0), 'b-', label='E model', linewidth=2)
    if autocorr_model_i:
        ax.plot(lags, np.mean(autocorr_model_i, axis=0), 'r-', label='I model', linewidth=2)
    if autocorr_real_e:
        ax.plot(lags, np.mean(autocorr_real_e, axis=0), 'b--', label='E real', linewidth=2, alpha=0.5)
    if autocorr_real_i:
        ax.plot(lags, np.mean(autocorr_real_i, axis=0), 'r--', label='I real', linewidth=2, alpha=0.5)
    ax.set_xlabel('Lag (ms)')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Model vs Real: Autocorrelation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(analysis_dir / 'autocorrelation_analysis.png', dpi=150)
    plt.close()

    # 3. Signal-to-noise analysis
    print("Computing signal-to-noise analysis...")

    # Per-neuron correlations
    per_neuron_corr = compute_per_neuron_correlations(model, data, device)

    # Mean firing rates
    mean_rates = target_psth.mean(axis=0)

    # Variance of PSTH (signal)
    psth_var = target_psth.var(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Correlation vs mean firing rate
    ax = axes[0]
    ax.scatter(mean_rates[e_indices], per_neuron_corr[e_indices],
               c='blue', alpha=0.6, label='E neurons')
    ax.scatter(mean_rates[i_indices], per_neuron_corr[i_indices],
               c='red', alpha=0.6, label='I neurons')
    ax.set_xlabel('Mean Firing Rate (sp/s)')
    ax.set_ylabel('PSTH Correlation')
    ax.set_title('Fit Quality vs Firing Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Correlation vs PSTH variance (signal)
    ax = axes[1]
    ax.scatter(psth_var[e_indices], per_neuron_corr[e_indices],
               c='blue', alpha=0.6, label='E neurons')
    ax.scatter(psth_var[i_indices], per_neuron_corr[i_indices],
               c='red', alpha=0.6, label='I neurons')
    ax.set_xlabel('PSTH Variance')
    ax.set_ylabel('PSTH Correlation')
    ax.set_title('Fit Quality vs Signal Strength')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(analysis_dir / 'signal_noise_analysis.png', dpi=150)
    plt.close()

    # Summary statistics
    summary = {
        'n_e_neurons': len(e_indices),
        'n_i_neurons': len(i_indices),
        'mean_corr_e': float(per_neuron_corr[e_indices].mean()) if len(e_indices) > 0 else 0,
        'mean_corr_i': float(per_neuron_corr[i_indices].mean()) if len(i_indices) > 0 else 0,
        'mean_rate_e': float(mean_rates[e_indices].mean()) if len(e_indices) > 0 else 0,
        'mean_rate_i': float(mean_rates[i_indices].mean()) if len(i_indices) > 0 else 0,
        'top_input_for_i': input_names[np.argmax(mean_corr_i)] if len(i_indices) > 0 else 'N/A',
    }

    return summary


def analyze_power_spectra(
    model: EIRNN,
    data: dict,
    neuron_info: dict,
    bin_size_ms: float,
    device: str,
    output_dir: Path
):
    """
    Power spectrum analysis of real vs model PSTHs.
    """
    model.eval()
    analysis_dir = output_dir / 'analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        inputs = data['inputs'].to(device)
        targets = data['targets'].to(device)

        model_rates, _ = model(inputs)

        # Get PSTHs
        model_psth = model_rates.mean(dim=0).cpu().numpy()
        target_psth = targets.mean(dim=0).cpu().numpy()

    n_recorded = target_psth.shape[1]
    n_exc = neuron_info['n_exc']

    # Sampling frequency
    fs = 1000.0 / bin_size_ms  # Hz

    # Compute power spectra
    def compute_psd(x, fs):
        """Compute power spectral density using Welch's method."""
        f, psd = signal.welch(x, fs=fs, nperseg=min(64, len(x)))
        return f, psd

    # Compute for all neurons
    psd_real_all = []
    psd_model_all = []
    freq = None

    for i in range(n_recorded):
        f, psd_real = compute_psd(target_psth[:, i], fs)
        _, psd_model = compute_psd(model_psth[:, i], fs)
        psd_real_all.append(psd_real)
        psd_model_all.append(psd_model)
        freq = f

    psd_real_all = np.array(psd_real_all)
    psd_model_all = np.array(psd_model_all)

    # Separate E and I
    e_indices = np.arange(min(n_exc, n_recorded))
    i_indices = np.arange(n_exc, n_recorded)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # All neurons - mean PSD
    ax = axes[0, 0]
    ax.semilogy(freq, psd_real_all.mean(axis=0), 'b-', label='Real', linewidth=2)
    ax.semilogy(freq, psd_model_all.mean(axis=0), 'r-', label='Model', linewidth=2)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density')
    ax.set_title('Mean Power Spectrum (All Neurons)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # E neurons
    ax = axes[0, 1]
    if len(e_indices) > 0:
        ax.semilogy(freq, psd_real_all[e_indices].mean(axis=0), 'b-', label='Real', linewidth=2)
        ax.semilogy(freq, psd_model_all[e_indices].mean(axis=0), 'r-', label='Model', linewidth=2)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density')
    ax.set_title('Mean Power Spectrum (E Neurons)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # I neurons
    ax = axes[1, 0]
    if len(i_indices) > 0:
        ax.semilogy(freq, psd_real_all[i_indices].mean(axis=0), 'b-', label='Real', linewidth=2)
        ax.semilogy(freq, psd_model_all[i_indices].mean(axis=0), 'r-', label='Model', linewidth=2)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density')
    ax.set_title('Mean Power Spectrum (I Neurons)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Power ratio (model/real) vs frequency
    ax = axes[1, 1]
    power_ratio = psd_model_all.mean(axis=0) / (psd_real_all.mean(axis=0) + 1e-10)
    ax.plot(freq, power_ratio, 'k-', linewidth=2)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Ratio (Model/Real)')
    ax.set_title('Power Ratio vs Frequency')
    ax.set_ylim([0, 2])
    ax.grid(True, alpha=0.3)

    # Find cutoff frequency (where ratio drops below 0.5)
    cutoff_idx = np.where(power_ratio < 0.5)[0]
    if len(cutoff_idx) > 0:
        cutoff_freq = freq[cutoff_idx[0]]
        ax.axvline(x=cutoff_freq, color='red', linestyle='--', alpha=0.5)
        ax.text(cutoff_freq, 1.5, f'Cutoff: {cutoff_freq:.1f} Hz', rotation=90, va='bottom')

    plt.tight_layout()
    plt.savefig(analysis_dir / 'power_spectra_comparison.png', dpi=150)
    plt.close()

    # Summary
    summary = {
        'mean_power_ratio': float(power_ratio.mean()),
        'low_freq_ratio': float(power_ratio[:len(power_ratio)//3].mean()),
        'high_freq_ratio': float(power_ratio[len(power_ratio)//3:].mean()),
    }

    return summary


def analyze_ei_weights(
    model: EIRNN,
    neuron_info: dict,
    output_dir: Path
):
    """
    Analyze learned weights in bypass_dale model.
    """
    analysis_dir = output_dir / 'analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)

    if not model.bypass_dale:
        print("Model has Dale's law enforced, skipping E/I weight analysis")
        return {}

    # Get raw weights
    W_rec = model.W_rec.detach().cpu().numpy()
    n_recorded = neuron_info['n_exc'] + neuron_info['n_inh']

    # Compute net output per neuron (sum of outgoing weights)
    net_output = W_rec.sum(axis=0)[:n_recorded]  # Column sums

    # Original E/I labels
    original_e = np.arange(neuron_info['n_exc'])
    original_i = np.arange(neuron_info['n_exc'], n_recorded)

    # Classify based on learned weights
    learned_e = net_output > 0
    learned_i = net_output <= 0

    # Compute agreement
    n_correct_e = np.sum(learned_e[original_e])
    n_correct_i = np.sum(learned_i[original_i])
    agreement = (n_correct_e + n_correct_i) / n_recorded

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(original_e, net_output[original_e], c='blue', alpha=0.7, label='Original E')
    ax.scatter(original_i, net_output[original_i], c='red', alpha=0.7, label='Original I')
    ax.axhline(y=0, color='gray', linestyle='--')
    ax.set_xlabel('Neuron Index')
    ax.set_ylabel('Net Output Weight')
    ax.set_title('Learned Net Output by Original E/I Label')
    ax.legend()

    ax = axes[1]
    ax.hist(net_output[original_e], bins=20, alpha=0.5, label='Original E', color='blue')
    ax.hist(net_output[original_i], bins=20, alpha=0.5, label='Original I', color='red')
    ax.axvline(x=0, color='gray', linestyle='--')
    ax.set_xlabel('Net Output Weight')
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of Net Output (Agreement: {agreement:.1%})')
    ax.legend()

    plt.tight_layout()
    plt.savefig(analysis_dir / 'ei_assignment_analysis.png', dpi=150)
    plt.close()

    return {
        'agreement_with_original': float(agreement),
        'n_e_classified_as_i': int(np.sum(~learned_e[original_e])),
        'n_i_classified_as_e': int(np.sum(~learned_i[original_i])),
    }


# ============================================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================================

def get_experiments():
    """Define all Phase 5 experiments."""

    base_config = {
        'input_embed_dim': 56,
        'input_embed_type': 'attention',
        'attention_heads': 4,
        'learnable_h0': True,
        'h0_init': 0.1,
        'use_grad_balancing': True,
        'ltrial_scale': 0.5,
    }

    experiments = {
        # Experiment 0: Extended training
        'extended_training': {
            **base_config,
            'name': 'extended_training',
            'max_epochs': 1000,
            'patience': 150,
        },

        # Experiment 1: Faster dynamics
        'fast_tau_25': {
            **base_config,
            'name': 'fast_tau_25',
            'tau': 25.0,
            'max_epochs': 500,
            'patience': 100,
        },

        # Experiment 2a: Higher noise (0.3)
        'noise_0.3': {
            **base_config,
            'name': 'noise_0.3',
            'noise_scale': 0.3,
            'max_epochs': 500,
            'patience': 100,
        },

        # Experiment 2b: Higher noise (0.5)
        'noise_0.5': {
            **base_config,
            'name': 'noise_0.5',
            'noise_scale': 0.5,
            'max_epochs': 500,
            'patience': 100,
        },

        # Experiment 3: Bypass Dale's law
        'bypass_dale': {
            **base_config,
            'name': 'bypass_dale',
            'bypass_dale': True,
            'max_epochs': 500,
            'patience': 100,
        },

        # Experiment 4: Learnable tau (population)
        'learnable_tau': {
            **base_config,
            'name': 'learnable_tau',
            'learnable_tau': 'population',
            'tau_e_init': 50.0,
            'tau_i_init': 35.0,
            'max_epochs': 500,
            'patience': 100,
        },

        # Experiment 5: Poisson L_trial
        'poisson_ltrial': {
            **base_config,
            'name': 'poisson_ltrial',
            'use_poisson_ltrial': True,
            'max_epochs': 500,
            'patience': 100,
        },

        # Experiment 6: Low-rank connectivity
        'low_rank_50': {
            **base_config,
            'name': 'low_rank_50',
            'low_rank': 50,
            'max_epochs': 500,
            'patience': 100,
        },
    }

    return experiments


def run_experiments(
    data_path: str,
    output_dir: str,
    device: str = 'cpu',
    seed: int = 42,
    experiments_to_run: list = None
):
    """Run Phase 5 experiments."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    dataset = load_session(data_path)
    train_idx, val_idx = train_val_split(dataset, val_fraction=0.2, seed=seed)

    all_data = dataset.get_all_trials()
    train_data = {
        'inputs': all_data['inputs'][train_idx],
        'targets': all_data['targets'][train_idx],
        'mask': all_data['mask'][train_idx],
    }
    val_data = {
        'inputs': all_data['inputs'][val_idx],
        'targets': all_data['targets'][val_idx],
        'mask': all_data['mask'][val_idx],
    }

    neuron_info = dataset.get_neuron_info()
    n_inputs = dataset.get_input_dim()
    bin_size_ms = dataset.bin_size_ms

    print(f"Train trials: {len(train_idx)}, Val trials: {len(val_idx)}")
    print(f"Input dim: {n_inputs}, Bin size: {bin_size_ms}ms")

    # Reference values
    phase4_best = 0.4021

    # Get experiments
    all_experiments = get_experiments()

    # Filter experiments if specified
    if experiments_to_run:
        experiments = {k: v for k, v in all_experiments.items() if k in experiments_to_run}
    else:
        experiments = all_experiments

    # Run experiments
    results = {}
    for name, config in experiments.items():
        print(f"\n{'='*60}")
        print(f"Running experiment: {name}")
        print(f"{'='*60}")

        try:
            result = train_with_config(
                config=config,
                train_data=train_data,
                val_data=val_data,
                neuron_info=neuron_info,
                n_inputs=n_inputs,
                bin_size_ms=bin_size_ms,
                device=device,
                seed=seed
            )

            results[name] = result

            # Save intermediate results
            result_path = output_dir / f"{name}_result.json"
            result_json = {
                'best_val_corr': result['best_val_corr'],
                'final_val_corr': result['final_val_corr'],
                'epochs_trained': result['epochs_trained'],
                'tau_values': result['tau_values'],
                'alpha_values': result['alpha_values'],
                'h0_values': result['h0_values'],
                'config': result['config']
            }
            with open(result_path, 'w') as f:
                json.dump(result_json, f, indent=2)

            # Save best model
            if result.get('best_model_state') is not None:
                model_path = output_dir / f"{name}_model_best.pt"
                torch.save(result['best_model_state'], model_path)

            print(f"Best val correlation: {result['best_val_corr']:.4f}")

            # Run analysis for bypass_dale experiment
            if name == 'bypass_dale' and result.get('best_model_state') is not None:
                print("Running E/I weight analysis...")
                model = create_model_from_data(
                    n_classic=neuron_info['n_exc'],
                    n_interneuron=neuron_info['n_inh'],
                    n_inputs=n_inputs,
                    bypass_dale=True,
                    input_embed_dim=56,
                    input_embed_type='attention',
                    learnable_h0=True,
                    device=device
                )
                model.load_state_dict(result['best_model_state'])
                ei_analysis = analyze_ei_weights(model, neuron_info, output_dir)
                result['ei_analysis'] = ei_analysis

        except Exception as e:
            print(f"ERROR in experiment {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = {'best_val_corr': float('-inf'), 'error': str(e)}

    # Run analysis on best model
    print("\n" + "="*60)
    print("RUNNING ANALYTICAL DEEP-DIVES")
    print("="*60)

    # Find best model
    best_name = max(
        [k for k, v in results.items() if 'error' not in v],
        key=lambda k: results[k]['best_val_corr']
    )
    best_result = results[best_name]

    if best_result.get('best_model_state') is not None:
        print(f"Using best model: {best_name} (val_corr={best_result['best_val_corr']:.4f})")

        # Recreate model
        model = create_model_from_data(
            n_classic=neuron_info['n_exc'],
            n_interneuron=neuron_info['n_inh'],
            n_inputs=n_inputs,
            bypass_dale=best_result['config'].get('bypass_dale', False),
            low_rank=best_result['config'].get('low_rank', None),
            input_embed_dim=56,
            input_embed_type='attention',
            learnable_h0=True,
            device=device
        )
        model.load_state_dict(best_result['best_model_state'])

        # I neuron analysis
        print("\nAnalyzing I neurons...")
        i_neuron_summary = analyze_i_neurons(model, val_data, neuron_info, device, output_dir)

        # Power spectra analysis
        print("\nAnalyzing power spectra...")
        spectra_summary = analyze_power_spectra(model, val_data, neuron_info, bin_size_ms, device, output_dir)

        # Save analysis summaries
        analysis_summary = {
            'best_model': best_name,
            'best_val_corr': best_result['best_val_corr'],
            'i_neuron_analysis': i_neuron_summary,
            'spectra_analysis': spectra_summary,
        }
        with open(output_dir / 'analysis' / 'analysis_summary.json', 'w') as f:
            json.dump(analysis_summary, f, indent=2)

    # Summary
    print("\n" + "="*60)
    print("PHASE 5 EXPERIMENT SUMMARY")
    print("="*60)
    print(f"{'Configuration':<25} {'Best Val Corr':>15} {'vs Phase4':>12}")
    print("-"*55)

    sorted_results = sorted(
        [(k, v) for k, v in results.items() if 'error' not in v],
        key=lambda x: x[1]['best_val_corr'],
        reverse=True
    )
    for name, result in sorted_results:
        diff = result['best_val_corr'] - phase4_best
        marker = " *" if diff > 0 else ""
        print(f"{name:<25} {result['best_val_corr']:>15.4f} {diff:>+11.4f}{marker}")

    print("\n" + "-"*55)
    print(f"Phase 4 best (reference): {phase4_best:.4f}")

    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'data_path': str(data_path),
        'phase4_best': phase4_best,
        'results': {
            name: {
                'best_val_corr': r.get('best_val_corr', float('-inf')),
                'final_val_corr': r.get('final_val_corr'),
                'epochs_trained': r.get('epochs_trained'),
                'tau_values': r.get('tau_values', {}),
                'alpha_values': r.get('alpha_values', {}),
                'h0_values': r.get('h0_values', {}),
                'error': r.get('error')
            }
            for name, r in results.items()
        }
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Plot comparison
    plot_experiment_comparison(results, output_dir / 'analysis' / 'experiment_comparison.png', phase4_best)

    return results


def plot_experiment_comparison(results: dict, save_path: str, phase4_best: float):
    """Plot comparison of all experiments."""
    valid_results = {k: v for k, v in results.items() if 'error' not in v}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart
    ax = axes[0]
    names = list(valid_results.keys())
    corrs = [valid_results[n]['best_val_corr'] for n in names]

    sorted_idx = np.argsort(corrs)[::-1]
    names = [names[i] for i in sorted_idx]
    corrs = [corrs[i] for i in sorted_idx]

    colors = ['green' if c > phase4_best else 'steelblue' for c in corrs]

    bars = ax.barh(names, corrs, color=colors)
    ax.set_xlabel('Best Validation PSTH Correlation')
    ax.set_title('Phase 5 Experiment Comparison')
    ax.axvline(x=phase4_best, color='orange', linestyle='--', label=f'Phase 4 Best ({phase4_best:.4f})')
    ax.legend(fontsize=8)
    ax.set_xlim(min(min(corrs) - 0.02, 0.35), max(max(corrs) + 0.02, phase4_best + 0.02))

    # Training curves
    ax = axes[1]
    for name in names[:5]:
        if 'history' in valid_results[name]:
            history = valid_results[name]['history']
            ax.plot(history['val_corr'], label=f"{name} ({valid_results[name]['best_val_corr']:.3f})")

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation PSTH Correlation')
    ax.set_title('Training Curves')
    ax.axhline(y=phase4_best, color='orange', linestyle='--', alpha=0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Ensure directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved comparison plot to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Phase 5 experiments - Limitation Analysis')
    parser.add_argument('--data', type=str, required=True, help='Path to .mat file')
    parser.add_argument('--output', type=str, default='results/phase5/', help='Output directory')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--experiments', type=str, nargs='+', default=None,
                        help='Specific experiments to run (default: all)')

    args = parser.parse_args()

    run_experiments(
        data_path=args.data,
        output_dir=args.output,
        device=args.device,
        seed=args.seed,
        experiments_to_run=args.experiments
    )


if __name__ == "__main__":
    main()
