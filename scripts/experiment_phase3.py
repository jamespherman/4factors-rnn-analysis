"""
Phase 3 Experiment Script - Extended embedding exploration.

Tests improvements from specs/improvement_plan_phase3.md:
1. Phase 3a: Embedding Exploration (larger embeddings, deep embeddings)
2. Phase 3b: Embedding + Single Improvement (alpha init, lr, optimizer)
3. Phase 3c: Architecture Variants (typed, recurrent, attention)

Usage:
    python scripts/experiment_phase3.py --data data/rnn_export_Newton_08_15_2025_SC.mat --output results/phase3/
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

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import EIRNN, create_model_from_data
from src.losses import (
    EIRNNLoss, compute_L_neuron, compute_L_poisson, compute_L_neuron_hybrid,
    compute_activity_regularization, compute_L_trial, compute_L_reg,
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
    """
    Train a model with the given configuration.

    Args:
        config: Dict with training configuration
        train_data: Training data dict
        val_data: Validation data dict
        neuron_info: Dict with n_exc, n_inh
        n_inputs: Number of input dimensions
        bin_size_ms: Bin size in ms
        device: Device to use
        seed: Random seed

    Returns:
        Dict with training results
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create model
    model = create_model_from_data(
        n_classic=neuron_info['n_exc'],
        n_interneuron=neuron_info['n_inh'],
        n_inputs=n_inputs,
        enforce_ratio=True,
        dt=bin_size_ms,
        noise_scale=config.get('noise_scale', 0.1),
        learnable_tau=config.get('learnable_tau', 'none'),
        tau_e_init=config.get('tau_e_init', 50.0),
        tau_i_init=config.get('tau_i_init', 35.0),
        learnable_alpha=config.get('learnable_alpha', 'none'),
        alpha_init=config.get('alpha_init', 0.5),
        alpha_e_init=config.get('alpha_e_init', None),
        alpha_i_init=config.get('alpha_i_init', None),
        low_rank=config.get('low_rank', None),
        input_embed_dim=config.get('input_embed_dim', None),
        input_embed_type=config.get('input_embed_type', 'learnable'),
        input_time_lags=config.get('input_time_lags', None),
        embed_hidden_dim=config.get('embed_hidden_dim', None),
        embed_n_layers=config.get('embed_n_layers', 2),
        input_groups=config.get('input_groups', None),
        group_embed_dims=config.get('group_embed_dims', None),
        gru_hidden_dim=config.get('gru_hidden_dim', None),
        attention_heads=config.get('attention_heads', 4),
        device=device
    )

    # Print model info
    name = config.get('name', 'unknown')
    if config.get('low_rank') is not None:
        print(f"  [{name}] Low-rank W_rec: rank={config['low_rank']}")
    if config.get('learnable_alpha', 'none') != 'none':
        alpha_e = config.get('alpha_e_init', config.get('alpha_init', 0.5))
        alpha_i = config.get('alpha_i_init')
        print(f"  [{name}] Learnable alpha: mode={config['learnable_alpha']}, E_init={alpha_e}, I_init={alpha_i}")
    if config.get('learnable_tau', 'none') != 'none':
        print(f"  [{name}] Learnable tau: mode={config['learnable_tau']}, tau_e={config.get('tau_e_init')}, tau_i={config.get('tau_i_init')}")

    # Print embedding info
    embed_type = config.get('input_embed_type', 'learnable')
    if config.get('input_embed_dim') is not None:
        print(f"  [{name}] Input embedding: dim={config['input_embed_dim']}, type={embed_type}")
    if embed_type == 'deep':
        print(f"  [{name}] Deep embedding: hidden={config.get('embed_hidden_dim')}, layers={config.get('embed_n_layers', 2)}")
    elif embed_type == 'typed':
        print(f"  [{name}] Typed embedding: groups={config.get('input_groups')}")
    elif embed_type == 'recurrent':
        print(f"  [{name}] Recurrent embedding: GRU hidden={config.get('gru_hidden_dim')}")
    elif embed_type == 'attention':
        print(f"  [{name}] Attention embedding: heads={config.get('attention_heads', 4)}")

    # Optimizer
    lr = config.get('lr', 1e-3)
    optimizer_type = config.get('optimizer', 'adam')

    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == 'adamw':
        # AdamW with weight decay on embedding only
        embed_params = []
        other_params = []
        for name_p, param in model.named_parameters():
            if 'input_embed' in name_p:
                embed_params.append(param)
            else:
                other_params.append(param)

        weight_decay = config.get('weight_decay', 0.01)
        optimizer = torch.optim.AdamW([
            {'params': embed_params, 'weight_decay': weight_decay},
            {'params': other_params, 'weight_decay': 0.0}
        ], lr=lr)
        print(f"  [{name}] Using AdamW with weight_decay={weight_decay} on embedding")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Scheduler
    scheduler_type = config.get('scheduler', 'plateau')
    max_epochs = config.get('max_epochs', 500)

    if scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=50, min_lr=1e-5
        )
    elif scheduler_type == 'cosine':
        warmup_epochs = config.get('warmup_epochs', 50)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, warmup_epochs, max_epochs
        )
    else:
        scheduler = None

    # Loss configuration
    loss_type = config.get('loss_type', 'correlation')
    poisson_weight = config.get('poisson_weight', 0.5)
    use_activity_reg = config.get('use_activity_reg', False)
    lambda_activity = config.get('lambda_activity', 0.01)
    lambda_trial = config.get('lambda_trial', 0.5)
    lambda_reg = config.get('lambda_reg', 1e-4)
    use_poisson_ltrial = config.get('use_poisson_ltrial', False)

    # Gradient balancing parameters
    use_grad_balancing = config.get('use_grad_balancing', True)
    ltrial_scale = config.get('ltrial_scale', 0.5)

    # For gradient balancing, track EMAs
    loss_ema = {'L_neuron': 1.0, 'L_trial': 1.0}
    ema_decay = 0.99

    # Training loop
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
        'alpha_values': [],
        'tau_values': []
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

        # Compute losses based on loss_type
        if loss_type == 'correlation':
            L_neuron = compute_L_neuron(
                model_rates_recorded, targets, bin_size_ms,
                mask=mask, lambda_scale=0.1, lambda_var=0.05
            )
        elif loss_type == 'poisson':
            L_neuron = compute_L_poisson(
                model_rates_recorded, targets, bin_size_ms, mask=mask
            )
        elif loss_type == 'hybrid':
            L_neuron = compute_L_neuron_hybrid(
                model_rates_recorded, targets, bin_size_ms,
                mask=mask, poisson_weight=poisson_weight
            )
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        # L_trial with Sinkhorn (optionally with Poisson distance)
        L_trial_raw = compute_L_trial(
            model_rates_recorded, targets, bin_size_ms,
            mask=mask, sinkhorn_iters=20, sinkhorn_epsilon=0.1,
            use_poisson_distance=use_poisson_ltrial
        )

        # L_reg
        L_reg = compute_L_reg(model, lambda_reg)

        # Activity regularization
        if use_activity_reg:
            L_activity = compute_activity_regularization(
                model_rates_recorded, lambda_mean=lambda_activity
            )
        else:
            L_activity = torch.tensor(0.0, device=device)

        # Gradient balancing
        if use_grad_balancing:
            # Update EMAs
            with torch.no_grad():
                loss_ema['L_neuron'] = ema_decay * loss_ema['L_neuron'] + (1 - ema_decay) * L_neuron.item()
                loss_ema['L_trial'] = ema_decay * loss_ema['L_trial'] + (1 - ema_decay) * L_trial_raw.item()

            # Normalize and combine
            L_neuron_norm = L_neuron / (loss_ema['L_neuron'] + 1e-8)
            L_trial_norm = L_trial_raw / (loss_ema['L_trial'] + 1e-8)
            loss = L_neuron_norm + ltrial_scale * L_trial_norm + L_reg + L_activity
        else:
            loss = L_neuron + lambda_trial * L_trial_raw + L_reg + L_activity

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss = loss.item()
        train_corr = compute_psth_correlation(model, train_data, device)

        # Validate
        val_loss_value, val_corr = 0.0, 0.0
        model.eval()
        with torch.no_grad():
            inputs_val = val_data['inputs'].to(device)
            targets_val = val_data['targets'].to(device)
            mask_val = val_data['mask'].to(device)

            model_rates_val, _ = model(inputs_val)
            model_rates_val_recorded = model_rates_val[:, :, :n_recorded]

            if loss_type == 'correlation':
                L_neuron_val = compute_L_neuron(
                    model_rates_val_recorded, targets_val, bin_size_ms,
                    mask=mask_val, lambda_scale=0.1, lambda_var=0.05
                )
            elif loss_type == 'poisson':
                L_neuron_val = compute_L_poisson(
                    model_rates_val_recorded, targets_val, bin_size_ms, mask=mask_val
                )
            elif loss_type == 'hybrid':
                L_neuron_val = compute_L_neuron_hybrid(
                    model_rates_val_recorded, targets_val, bin_size_ms,
                    mask=mask_val, poisson_weight=poisson_weight
                )

            val_loss_value = L_neuron_val.item()
            val_corr = compute_psth_correlation(model, val_data, device)

        # Update scheduler
        if scheduler_type == 'plateau':
            scheduler.step(val_loss_value)
        elif scheduler_type == 'cosine':
            scheduler.step()

        # Record history
        history['train_corr'].append(train_corr)
        history['val_corr'].append(val_corr)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss_value)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Record alpha/tau values if learnable
        if hasattr(model, 'get_alpha_values'):
            history['alpha_values'].append(model.get_alpha_values())
        if hasattr(model, 'get_tau_values'):
            history['tau_values'].append(model.get_tau_values())

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
            best_model_state = model.state_dict().copy()
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Get final tau/alpha values
    tau_values = model.get_tau_values() if hasattr(model, 'get_tau_values') else {}
    alpha_values = model.get_alpha_values() if hasattr(model, 'get_alpha_values') else {}

    return {
        'best_val_corr': best_val_corr,
        'final_val_corr': val_corr,
        'history': history,
        'epochs_trained': epoch + 1,
        'tau_values': tau_values,
        'alpha_values': alpha_values,
        'config': config,
        'best_model_state': best_model_state
    }


def run_experiments(
    data_path: str,
    output_dir: str,
    device: str = 'cpu',
    seed: int = 42,
    experiments_to_run: list = None
):
    """Run all Phase 3 improvement experiments."""
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

    # Define all experiment configurations
    # Phase 3 priority ordering from improvement_plan_phase3.md
    all_experiments = {
        # ========================================
        # PHASE 3a: Embedding Exploration (High Priority)
        # ========================================

        # 8x expansion (14 -> 112) - larger than phase 2c best
        'embed_112_learnable': {
            'name': 'embed_112_learnable',
            'loss_type': 'correlation',
            'input_embed_dim': 112,
            'input_embed_type': 'learnable',
            'scheduler': 'plateau',
            'use_grad_balancing': True,
            'ltrial_scale': 0.5,
            'max_epochs': 500,
            'patience': 100
        },

        # 12x expansion (14 -> 168)
        'embed_168_learnable': {
            'name': 'embed_168_learnable',
            'loss_type': 'correlation',
            'input_embed_dim': 168,
            'input_embed_type': 'learnable',
            'scheduler': 'plateau',
            'use_grad_balancing': True,
            'ltrial_scale': 0.5,
            'max_epochs': 500,
            'patience': 100
        },

        # Deep embedding: 2-layer (14 -> 28 -> 56)
        'deep_embed_56': {
            'name': 'deep_embed_56',
            'loss_type': 'correlation',
            'input_embed_dim': 56,
            'input_embed_type': 'deep',
            'embed_hidden_dim': 28,
            'embed_n_layers': 2,
            'scheduler': 'plateau',
            'use_grad_balancing': True,
            'ltrial_scale': 0.5,
            'max_epochs': 500,
            'patience': 100
        },

        # Higher noise with embedding (noise_scale=0.15)
        'embed_56_higher_noise': {
            'name': 'embed_56_higher_noise',
            'loss_type': 'correlation',
            'input_embed_dim': 56,
            'input_embed_type': 'learnable',
            'noise_scale': 0.15,
            'scheduler': 'plateau',
            'use_grad_balancing': True,
            'ltrial_scale': 0.5,
            'max_epochs': 500,
            'patience': 100
        },

        # ========================================
        # PHASE 3b: Embedding + Single Improvement (Medium Priority)
        # ========================================

        # Embedding + optimal alpha init (E: 0.54, I: 0.72)
        'embed_56_alpha_neuron_optimal': {
            'name': 'embed_56_alpha_neuron_optimal',
            'loss_type': 'correlation',
            'input_embed_dim': 56,
            'input_embed_type': 'learnable',
            'learnable_alpha': 'neuron',
            'alpha_e_init': 0.54,
            'alpha_i_init': 0.72,
            'scheduler': 'plateau',
            'use_grad_balancing': True,
            'ltrial_scale': 0.5,
            'max_epochs': 500,
            'patience': 100
        },

        # Embedding + lower learning rate
        'embed_56_lower_lr': {
            'name': 'embed_56_lower_lr',
            'loss_type': 'correlation',
            'input_embed_dim': 56,
            'input_embed_type': 'learnable',
            'lr': 0.5e-3,
            'scheduler': 'plateau',
            'use_grad_balancing': True,
            'ltrial_scale': 0.5,
            'max_epochs': 500,
            'patience': 100
        },

        # Embedding + AdamW with weight decay on embedding only
        'embed_56_adamw': {
            'name': 'embed_56_adamw',
            'loss_type': 'correlation',
            'input_embed_dim': 56,
            'input_embed_type': 'learnable',
            'optimizer': 'adamw',
            'weight_decay': 0.01,
            'scheduler': 'plateau',
            'use_grad_balancing': True,
            'ltrial_scale': 0.5,
            'max_epochs': 500,
            'patience': 100
        },

        # ========================================
        # PHASE 3c: Architecture Variants (Lower Priority)
        # ========================================

        # Typed embedding: separate embeddings for input groups
        # Default: split inputs into 2 groups (first 7, last 7)
        'typed_embedding': {
            'name': 'typed_embedding',
            'loss_type': 'correlation',
            'input_embed_dim': 56,  # Total output dim
            'input_embed_type': 'typed',
            'input_groups': [(0, 7), (7, 14)],  # Two groups of 7 inputs each
            'group_embed_dims': [28, 28],  # Each group embeds to 28 dims
            'scheduler': 'plateau',
            'use_grad_balancing': True,
            'ltrial_scale': 0.5,
            'max_epochs': 500,
            'patience': 100
        },

        # Recurrent embedding: GRU preprocessing
        'recurrent_embedding': {
            'name': 'recurrent_embedding',
            'loss_type': 'correlation',
            'input_embed_dim': 56,
            'input_embed_type': 'recurrent',
            'gru_hidden_dim': 32,
            'scheduler': 'plateau',
            'use_grad_balancing': True,
            'ltrial_scale': 0.5,
            'max_epochs': 500,
            'patience': 100
        },

        # Attention embedding: self-attention on inputs
        'attention_embedding': {
            'name': 'attention_embedding',
            'loss_type': 'correlation',
            'input_embed_dim': 56,
            'input_embed_type': 'attention',
            'attention_heads': 4,
            'scheduler': 'plateau',
            'use_grad_balancing': True,
            'ltrial_scale': 0.5,
            'max_epochs': 500,
            'patience': 100
        },
    }

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

            # Don't store model state in results dict (too large for JSON)
            result_for_storage = {k: v for k, v in result.items() if k != 'best_model_state'}
            results[name] = result

            # Save intermediate results
            result_path = output_dir / f"{name}_result.json"
            result_json = {
                'best_val_corr': result['best_val_corr'],
                'final_val_corr': result['final_val_corr'],
                'epochs_trained': result['epochs_trained'],
                'tau_values': result['tau_values'],
                'alpha_values': result['alpha_values'],
                'config': result['config']
            }
            with open(result_path, 'w') as f:
                json.dump(result_json, f, indent=2)

            # Save best model
            if result.get('best_model_state') is not None:
                model_path = output_dir / f"{name}_model_best.pt"
                torch.save(result['best_model_state'], model_path)

            print(f"Best val correlation: {result['best_val_corr']:.4f}")

        except Exception as e:
            print(f"ERROR in experiment {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = {'best_val_corr': float('-inf'), 'error': str(e)}

    # Summary
    print("\n" + "="*60)
    print("PHASE 3 EXPERIMENT SUMMARY")
    print("="*60)
    print(f"{'Configuration':<40} {'Best Val Corr':>15}")
    print("-"*55)

    # Reference points
    phase2c_best = 0.3762
    target = 0.40
    stretch_target = 0.42

    sorted_results = sorted(
        [(k, v) for k, v in results.items() if 'error' not in v],
        key=lambda x: x[1]['best_val_corr'],
        reverse=True
    )
    for name, result in sorted_results:
        marker = ""
        if result['best_val_corr'] >= stretch_target:
            marker = " *** STRETCH ***"
        elif result['best_val_corr'] >= target:
            marker = " ** TARGET **"
        elif result['best_val_corr'] > phase2c_best:
            marker = " *"
        print(f"{name:<40} {result['best_val_corr']:>15.4f}{marker}")

    # Print errors
    errors = [(k, v) for k, v in results.items() if 'error' in v]
    if errors:
        print("\nEXPERIMENTS WITH ERRORS:")
        for name, result in errors:
            print(f"  {name}: {result['error']}")

    # Print comparison to baseline
    print("\n" + "-"*55)
    print("Reference points:")
    print(f"  Phase 2c best (embed_56_learnable):    {phase2c_best:.4f}")
    print(f"  Target:                                {target:.4f}")
    print(f"  Stretch target:                        {stretch_target:.4f}")

    if sorted_results:
        best_name, best_result = sorted_results[0]
        improvement = (best_result['best_val_corr'] - phase2c_best) / phase2c_best * 100
        gap_to_target = (target - best_result['best_val_corr']) / target * 100
        print(f"\nPhase 3 best ({best_name}):")
        print(f"  Best val corr: {best_result['best_val_corr']:.4f}")
        print(f"  vs Phase 2c:   {improvement:+.1f}%")
        if best_result['best_val_corr'] < target:
            print(f"  Gap to target: {gap_to_target:.1f}%")
        else:
            print(f"  Exceeded target by: {-gap_to_target:.1f}%")

    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'data_path': str(data_path),
        'phase2c_best': phase2c_best,
        'target': target,
        'stretch_target': stretch_target,
        'results': {
            name: {
                'best_val_corr': r.get('best_val_corr', float('-inf')),
                'final_val_corr': r.get('final_val_corr'),
                'epochs_trained': r.get('epochs_trained'),
                'tau_values': r.get('tau_values', {}),
                'alpha_values': r.get('alpha_values', {}),
                'error': r.get('error')
            }
            for name, r in results.items()
        }
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Plot comparison
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    if valid_results:
        plot_comparison(valid_results, output_dir / 'comparison.png',
                       phase2c_best=phase2c_best, target=target, stretch_target=stretch_target)

    return results


def plot_comparison(results: dict, save_path: str,
                    phase2c_best: float = 0.3762,
                    target: float = 0.40,
                    stretch_target: float = 0.42):
    """Plot comparison of all experiments."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Bar chart of best correlations
    ax = axes[0]
    names = list(results.keys())
    corrs = [results[n]['best_val_corr'] for n in names]

    # Sort by correlation
    sorted_idx = np.argsort(corrs)[::-1]
    names = [names[i] for i in sorted_idx]
    corrs = [corrs[i] for i in sorted_idx]

    # Color based on performance
    colors = []
    for c in corrs:
        if c >= stretch_target:
            colors.append('gold')  # Beat stretch target
        elif c >= target:
            colors.append('green')  # Beat target
        elif c > phase2c_best:
            colors.append('limegreen')  # Beat Phase 2c
        else:
            colors.append('steelblue')

    bars = ax.barh(names, corrs, color=colors)
    ax.set_xlabel('Best Validation PSTH Correlation')
    ax.set_title('Phase 3 Experiment Comparison')
    ax.axvline(x=stretch_target, color='gold', linestyle='--', label=f'Stretch ({stretch_target})')
    ax.axvline(x=target, color='green', linestyle='--', label=f'Target ({target})')
    ax.axvline(x=phase2c_best, color='orange', linestyle='--', label=f'Phase 2c Best ({phase2c_best})')
    ax.legend(fontsize=8)
    ax.set_xlim(min(min(corrs) - 0.02, 0.35), max(max(corrs) + 0.02, stretch_target + 0.02))

    # Training curves for top 5
    ax = axes[1]
    sorted_names = sorted(names, key=lambda n: results[n]['best_val_corr'], reverse=True)[:5]
    for name in sorted_names:
        if 'history' in results[name]:
            history = results[name]['history']
            ax.plot(history['val_corr'], label=f"{name} ({results[name]['best_val_corr']:.3f})")

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation PSTH Correlation')
    ax.set_title('Training Curves (Top 5)')
    ax.axhline(y=stretch_target, color='gold', linestyle='--', alpha=0.5)
    ax.axhline(y=target, color='green', linestyle='--', alpha=0.5)
    ax.axhline(y=phase2c_best, color='orange', linestyle='--', alpha=0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved comparison plot to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Phase 3 improvement experiments')
    parser.add_argument('--data', type=str, required=True, help='Path to .mat file')
    parser.add_argument('--output', type=str, default='results/phase3/',
                        help='Output directory')
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
