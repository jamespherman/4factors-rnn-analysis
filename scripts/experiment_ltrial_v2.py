"""
Follow-up experiments for L_trial optimization.

Building on the success of gradient balancing, explore:
1. Different epsilon values with gradient balancing
2. Gradient balancing with partial L_trial contribution
3. Warmup strategies
"""

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import create_model_from_data
from src.losses import compute_L_neuron, compute_L_trial, compute_L_reg
from src.data_loader import load_session, train_val_split


def compute_psth_correlation(model, data, device):
    model.eval()
    with torch.no_grad():
        inputs = data['inputs'].to(device)
        targets = data['targets'].to(device)
        model_rates, _ = model(inputs)
        n_recorded = targets.shape[2]
        model_psth = model_rates[:, :, :n_recorded].mean(dim=0).cpu().numpy()
        target_psth = targets.mean(dim=0).cpu().numpy()
        correlations = []
        for i in range(n_recorded):
            r = np.corrcoef(model_psth[:, i], target_psth[:, i])[0, 1]
            if not np.isnan(r):
                correlations.append(r)
        return np.mean(correlations) if correlations else 0.0


def train_gradient_balanced(
    data_path: str,
    output_dir: str,
    sinkhorn_epsilon: float = 0.1,
    ltrial_scale: float = 1.0,  # Scale L_trial gradient contribution
    warmup_epochs: int = 0,  # Epochs of L_neuron only before adding L_trial
    max_epochs: int = 500,
    patience: int = 100,
    device: str = 'cpu',
    seed: int = 42,
):
    """Train with gradient balancing and configurable options."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    dataset = load_session(data_path, validate=False)
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

    # Create model
    neuron_info = dataset.get_neuron_info()
    n_inputs = dataset.get_input_dim()

    model = create_model_from_data(
        n_classic=neuron_info['n_exc'],
        n_interneuron=neuron_info['n_inh'],
        n_inputs=n_inputs,
        dt=float(dataset.bin_size_ms),
        device=device
    )

    bin_size_ms = dataset.bin_size_ms
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=50, min_lr=1e-5
    )

    history = {'train_L_neuron': [], 'train_L_trial': [], 'val_psth_corr': []}
    best_val_corr = float('-inf')
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    config_str = f"eps={sinkhorn_epsilon},scale={ltrial_scale},warmup={warmup_epochs}"
    pbar = tqdm(range(max_epochs), desc=config_str)

    for epoch in pbar:
        model.train()

        inputs = train_data['inputs'].to(device)
        targets = train_data['targets'].to(device)
        mask = train_data['mask'].to(device)

        model_rates, _ = model(inputs)
        n_recorded = targets.shape[2]

        L_neuron = compute_L_neuron(
            model_rates[:, :, :n_recorded], targets,
            bin_size_ms=bin_size_ms, mask=mask
        )

        L_trial = compute_L_trial(
            model_rates[:, :, :n_recorded], targets,
            bin_size_ms=bin_size_ms, mask=mask,
            sinkhorn_epsilon=sinkhorn_epsilon
        )

        L_reg = compute_L_reg(model, lambda_l2=1e-4)

        optimizer.zero_grad()

        if epoch < warmup_epochs:
            # Warmup: L_neuron only
            total_loss = L_neuron + L_reg
            total_loss.backward()
        else:
            # Gradient balancing
            L_neuron.backward(retain_graph=True)
            grad_norm_neuron = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))

            optimizer.zero_grad()
            L_trial.backward(retain_graph=True)
            grad_norm_trial = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))

            optimizer.zero_grad()
            balanced_loss = (
                L_neuron / (grad_norm_neuron + 1e-8) +
                ltrial_scale * L_trial / (grad_norm_trial + 1e-8) +
                L_reg
            )
            balanced_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        val_psth_corr = compute_psth_correlation(model, val_data, device)
        scheduler.step(val_psth_corr)

        history['train_L_neuron'].append(L_neuron.item())
        history['train_L_trial'].append(L_trial.item())
        history['val_psth_corr'].append(val_psth_corr)

        pbar.set_postfix({'corr': f"{val_psth_corr:.3f}", 'L_n': f"{L_neuron.item():.3f}"})

        if val_psth_corr > best_val_corr:
            best_val_corr = val_psth_corr
            epochs_without_improvement = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    # Save best model at end
    if best_state is not None:
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': best_state,
            'val_psth_corr': best_val_corr,
        }, str(output_dir / 'model_best.pt'))

    results = {
        'config': {
            'sinkhorn_epsilon': sinkhorn_epsilon,
            'ltrial_scale': ltrial_scale,
            'warmup_epochs': warmup_epochs,
        },
        'best_val_corr': best_val_corr,
        'final_epoch': epoch,
        'history': history,
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return best_val_corr


def main():
    data_path = 'data/rnn_export_Newton_08_15_2025_SC.mat'
    base_dir = Path('results/ltrial_experiments_v2')
    base_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        # Vary epsilon with gradient balancing
        {'name': 'gradbal_eps_0.05', 'sinkhorn_epsilon': 0.05},
        {'name': 'gradbal_eps_0.2', 'sinkhorn_epsilon': 0.2},

        # Vary L_trial contribution scale
        {'name': 'gradbal_scale_0.25', 'sinkhorn_epsilon': 0.1, 'ltrial_scale': 0.25},
        {'name': 'gradbal_scale_0.5', 'sinkhorn_epsilon': 0.1, 'ltrial_scale': 0.5},
        {'name': 'gradbal_scale_2.0', 'sinkhorn_epsilon': 0.1, 'ltrial_scale': 2.0},

        # Warmup strategies
        {'name': 'gradbal_warmup_50', 'sinkhorn_epsilon': 0.1, 'warmup_epochs': 50},
        {'name': 'gradbal_warmup_100', 'sinkhorn_epsilon': 0.1, 'warmup_epochs': 100},

        # Combined: best epsilon + scale variations
        {'name': 'gradbal_eps_0.05_scale_0.5', 'sinkhorn_epsilon': 0.05, 'ltrial_scale': 0.5},
        {'name': 'gradbal_eps_0.2_scale_0.5', 'sinkhorn_epsilon': 0.2, 'ltrial_scale': 0.5},
    ]

    results_summary = []

    print("="*70)
    print("EXPERIMENT GRID V2: Gradient Balancing Refinements")
    print("="*70)

    for i, exp in enumerate(experiments):
        name = exp.pop('name')
        print(f"\n[{i+1}/{len(experiments)}] {name}: {exp}")

        best_corr = train_gradient_balanced(
            data_path=data_path,
            output_dir=str(base_dir / name),
            **exp
        )

        results_summary.append({'name': name, 'config': exp, 'best_val_corr': best_corr})
        print(f"  Result: {best_corr:.4f}")
        exp['name'] = name

    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY V2")
    print("="*70)

    results_summary.sort(key=lambda x: x['best_val_corr'], reverse=True)
    print(f"{'Experiment':<35} {'Best Corr':>10}")
    print("-"*50)
    for r in results_summary:
        print(f"{r['name']:<35} {r['best_val_corr']:>10.4f}")

    # Include previous best for comparison
    print("-"*50)
    print(f"{'(prev) gradbal_eps_0.1':<35} {'0.3434':>10}")
    print(f"{'(prev) baseline_no_ltrial':<35} {'0.3299':>10}")

    with open(base_dir / 'summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)


if __name__ == "__main__":
    main()
