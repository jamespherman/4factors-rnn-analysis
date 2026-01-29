"""
Experiment: Learnable Time Constants

Tests whether learnable tau (per-population or per-neuron) improves fitting.
"""

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import EIRNN, create_model_from_data
from src.model_extended import EIRNN_LearnableTau, create_model_from_data_extended
from src.losses import compute_L_neuron, compute_L_reg
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


def train_with_tau_config(
    data_path: str,
    output_dir: str,
    tau_mode: str = 'fixed',
    tau_e_init: float = 50.0,
    tau_i_init: float = 20.0,
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
    n_inputs = dataset.get_input_dim()
    
    if tau_mode == 'fixed':
        model = create_model_from_data(
            n_classic=neuron_info['n_exc'], n_interneuron=neuron_info['n_inh'],
            n_inputs=n_inputs, dt=float(dataset.bin_size_ms), device=device
        )
    elif tau_mode == 'fixed_fast_i':
        model = create_model_from_data_extended(
            n_classic=neuron_info['n_exc'], n_interneuron=neuron_info['n_inh'],
            n_inputs=n_inputs, model_type='learnable_tau', tau_mode='per_pop',
            tau_e_init=tau_e_init, tau_i_init=tau_i_init,
            dt=float(dataset.bin_size_ms), device=device
        )
        model.tau_e_raw.requires_grad = False
        model.tau_i_raw.requires_grad = False
    elif tau_mode in ['per_pop', 'per_neuron']:
        model = create_model_from_data_extended(
            n_classic=neuron_info['n_exc'], n_interneuron=neuron_info['n_inh'],
            n_inputs=n_inputs, model_type='learnable_tau', tau_mode=tau_mode,
            tau_e_init=tau_e_init, tau_i_init=tau_i_init,
            dt=float(dataset.bin_size_ms), device=device
        )
    else:
        raise ValueError(f"Unknown tau_mode: {tau_mode}")
    
    bin_size_ms = dataset.bin_size_ms
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=50, min_lr=1e-5)
    
    history = {'train_loss': [], 'val_psth_corr': [], 'tau_e': [], 'tau_i': []}
    best_val_corr = float('-inf')
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    
    pbar = tqdm(range(max_epochs), desc=f"tau={tau_mode}")
    
    for epoch in pbar:
        model.train()
        inputs = train_data['inputs'].to(device)
        targets = train_data['targets'].to(device)
        mask = train_data['mask'].to(device)
        
        model_rates, _ = model(inputs)
        n_recorded = targets.shape[2]
        
        L_neuron = compute_L_neuron(model_rates[:, :, :n_recorded], targets, bin_size_ms=bin_size_ms, mask=mask)
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
        
        if hasattr(model, 'get_tau'):
            tau = model.get_tau().detach().cpu()
            history['tau_e'].append(tau[:model.n_exc].mean().item())
            history['tau_i'].append(tau[model.n_exc:].mean().item())
            pbar.set_postfix({'corr': f"{val_psth_corr:.3f}", 'τE': f"{history['tau_e'][-1]:.1f}", 'τI': f"{history['tau_i'][-1]:.1f}"})
        else:
            pbar.set_postfix({'corr': f"{val_psth_corr:.3f}"})
        
        if val_psth_corr > best_val_corr:
            best_val_corr = val_psth_corr
            epochs_without_improvement = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            break
    
    final_tau = {}
    if hasattr(model, 'get_tau'):
        tau = model.get_tau().detach().cpu()
        final_tau = {'tau_e_mean': tau[:model.n_exc].mean().item(), 'tau_i_mean': tau[model.n_exc:].mean().item()}
    
    if best_state is not None:
        torch.save({'epoch': best_epoch, 'model_state_dict': best_state, 'val_psth_corr': best_val_corr, 'final_tau': final_tau},
                   str(output_dir / 'model_best.pt'))
    
    results = {'config': {'tau_mode': tau_mode, 'tau_e_init': tau_e_init, 'tau_i_init': tau_i_init},
               'best_val_corr': best_val_corr, 'final_epoch': epoch, 'final_tau': final_tau, 'history': history}
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return best_val_corr, final_tau


def main():
    data_path = 'data/rnn_export_Newton_08_15_2025_SC.mat'
    base_dir = Path('results/tau_experiments')
    base_dir.mkdir(parents=True, exist_ok=True)
    
    experiments = [
        {'name': 'fixed_50', 'tau_mode': 'fixed'},
        {'name': 'fixed_fast_i', 'tau_mode': 'fixed_fast_i', 'tau_e_init': 50.0, 'tau_i_init': 20.0},
        {'name': 'per_pop_50_20', 'tau_mode': 'per_pop', 'tau_e_init': 50.0, 'tau_i_init': 20.0},
        {'name': 'per_pop_50_50', 'tau_mode': 'per_pop', 'tau_e_init': 50.0, 'tau_i_init': 50.0},
        {'name': 'per_neuron_50_20', 'tau_mode': 'per_neuron', 'tau_e_init': 50.0, 'tau_i_init': 20.0},
    ]
    
    results_summary = []
    print("="*70)
    print("EXPERIMENT: Learnable Time Constants")
    print("="*70)
    
    for i, exp in enumerate(experiments):
        name = exp.pop('name')
        print(f"\n[{i+1}/{len(experiments)}] {name}")
        best_corr, final_tau = train_with_tau_config(data_path=data_path, output_dir=str(base_dir / name), **exp)
        results_summary.append({'name': name, 'config': exp, 'best_val_corr': best_corr, 'final_tau': final_tau})
        print(f"  Result: {best_corr:.4f}")
        if final_tau:
            print(f"  Final tau: E={final_tau.get('tau_e_mean', 'N/A'):.1f}, I={final_tau.get('tau_i_mean', 'N/A'):.1f}")
        exp['name'] = name
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    results_summary.sort(key=lambda x: x['best_val_corr'], reverse=True)
    for r in results_summary:
        tau = r.get('final_tau', {})
        print(f"{r['name']:<25} {r['best_val_corr']:.4f}  τE={tau.get('tau_e_mean', 50):.1f} τI={tau.get('tau_i_mean', 50):.1f}")
    
    with open(base_dir / 'summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)


if __name__ == "__main__":
    main()
