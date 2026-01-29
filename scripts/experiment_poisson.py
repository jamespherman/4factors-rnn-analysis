"""
Experiment: Poisson Loss vs MSE Loss

Tests whether Poisson negative log-likelihood improves fitting
compared to MSE-based correlation loss.
"""

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import create_model_from_data
from src.losses import compute_L_neuron, compute_L_reg
from src.losses_extended import compute_L_neuron_poisson, compute_L_neuron_hybrid
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


def train_with_loss_type(
    data_path: str,
    output_dir: str,
    loss_type: str = 'correlation',
    poisson_weight: float = 0.5,
    max_epochs: int = 500,
    patience: int = 100,
    device: str = 'cpu',
    seed: int = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = load_session(data_path, validate=False)
    train_idx, val_idx = train_val_split(dataset, val_fraction=0.2, seed=seed)
    
    all_data = dataset.get_all_trials()
    train_data = {k: all_data[k][train_idx] for k in ['inputs', 'targets', 'mask']}
    val_data = {k: all_data[k][val_idx] for k in ['inputs', 'targets', 'mask']}
    
    neuron_info = dataset.get_neuron_info()
    model = create_model_from_data(
        n_classic=neuron_info['n_exc'],
        n_interneuron=neuron_info['n_inh'],
        n_inputs=dataset.get_input_dim(),
        dt=float(dataset.bin_size_ms),
        device=device
    )
    
    bin_size_ms = dataset.bin_size_ms
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=50, min_lr=1e-5
    )
    
    history = {'train_loss': [], 'val_psth_corr': []}
    best_val_corr = float('-inf')
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    
    pbar = tqdm(range(max_epochs), desc=f"loss={loss_type}")
    
    for epoch in pbar:
        model.train()
        
        inputs = train_data['inputs'].to(device)
        targets = train_data['targets'].to(device)
        mask = train_data['mask'].to(device)
        
        model_rates, _ = model(inputs)
        n_recorded = targets.shape[2]
        
        if loss_type == 'correlation':
            L_neuron = compute_L_neuron(model_rates[:, :, :n_recorded], targets, bin_size_ms=bin_size_ms, mask=mask)
        elif loss_type == 'poisson':
            L_neuron = compute_L_neuron_poisson(model_rates[:, :, :n_recorded], targets, mask=mask)
        elif loss_type == 'hybrid':
            L_neuron = compute_L_neuron_hybrid(
                model_rates[:, :, :n_recorded], targets, bin_size_ms=bin_size_ms, mask=mask,
                poisson_weight=poisson_weight, correlation_weight=1.0 - poisson_weight
            )
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        
        L_reg = compute_L_reg(model, lambda_l2=1e-4)
        total_loss = L_neuron + L_reg
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        val_psth_corr = compute_psth_correlation(model, val_data, device)
        scheduler.step(val_psth_corr)
        
        history['train_loss'].append(total_loss.item())
        history['val_psth_corr'].append(val_psth_corr)
        
        pbar.set_postfix({'corr': f"{val_psth_corr:.3f}", 'loss': f"{L_neuron.item():.3f}"})
        
        if val_psth_corr > best_val_corr:
            best_val_corr = val_psth_corr
            epochs_without_improvement = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            break
    
    if best_state is not None:
        torch.save({'epoch': best_epoch, 'model_state_dict': best_state, 'val_psth_corr': best_val_corr},
                   str(output_dir / 'model_best.pt'))
    
    results = {'config': {'loss_type': loss_type, 'poisson_weight': poisson_weight},
               'best_val_corr': best_val_corr, 'final_epoch': epoch, 'history': history}
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return best_val_corr


def main():
    data_path = 'data/rnn_export_Newton_08_15_2025_SC.mat'
    base_dir = Path('results/poisson_experiments')
    base_dir.mkdir(parents=True, exist_ok=True)
    
    experiments = [
        {'name': 'correlation_baseline', 'loss_type': 'correlation'},
        {'name': 'poisson_only', 'loss_type': 'poisson'},
        {'name': 'hybrid_0.3', 'loss_type': 'hybrid', 'poisson_weight': 0.3},
        {'name': 'hybrid_0.5', 'loss_type': 'hybrid', 'poisson_weight': 0.5},
        {'name': 'hybrid_0.7', 'loss_type': 'hybrid', 'poisson_weight': 0.7},
    ]
    
    results_summary = []
    print("="*70)
    print("EXPERIMENT: Poisson Loss Variants")
    print("="*70)
    
    for i, exp in enumerate(experiments):
        name = exp.pop('name')
        print(f"\n[{i+1}/{len(experiments)}] {name}: {exp}")
        best_corr = train_with_loss_type(data_path=data_path, output_dir=str(base_dir / name), **exp)
        results_summary.append({'name': name, 'config': exp, 'best_val_corr': best_corr})
        print(f"  Result: {best_corr:.4f}")
        exp['name'] = name
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    results_summary.sort(key=lambda x: x['best_val_corr'], reverse=True)
    for r in results_summary:
        print(f"{r['name']:<25} {r['best_val_corr']:.4f}")
    print("-"*40)
    print(f"{'(baseline) gradbal_scale_0.5':<25} {'0.3614'}")
    
    with open(base_dir / 'summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)


if __name__ == "__main__":
    main()
