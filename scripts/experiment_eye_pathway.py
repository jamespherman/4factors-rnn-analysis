"""
Experiment: Eye Position MLP Pathway
"""

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_extended import create_model_from_data_extended
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
        correlations = [np.corrcoef(model_psth[:, i], target_psth[:, i])[0, 1] for i in range(n_recorded)]
        return np.nanmean(correlations)


def train_eye_mlp(data_path, output_dir, eye_mlp_hidden=16, max_epochs=500, patience=100, device='cpu', seed=42):
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
    model = create_model_from_data_extended(
        n_classic=neuron_info['n_exc'], n_interneuron=neuron_info['n_inh'],
        n_inputs=dataset.get_input_dim(), model_type='eye_mlp', eye_mlp_hidden=eye_mlp_hidden,
        dt=float(dataset.bin_size_ms), device=device
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=50, min_lr=1e-5)
    
    history = {'train_loss': [], 'val_psth_corr': []}
    best_val_corr = float('-inf')
    best_state, best_epoch = None, 0
    epochs_without_improvement = 0
    
    for epoch in tqdm(range(max_epochs), desc=f"eye_mlp={eye_mlp_hidden}"):
        model.train()
        inputs, targets, mask = [train_data[k].to(device) for k in ['inputs', 'targets', 'mask']]
        
        model_rates, _ = model(inputs)
        L_neuron = compute_L_neuron(model_rates[:, :, :targets.shape[2]], targets, bin_size_ms=dataset.bin_size_ms, mask=mask)
        total_loss = L_neuron + compute_L_reg(model, lambda_l2=1e-4)
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        val_psth_corr = compute_psth_correlation(model, val_data, device)
        scheduler.step(val_psth_corr)
        
        history['train_loss'].append(total_loss.item())
        history['val_psth_corr'].append(val_psth_corr)
        
        if val_psth_corr > best_val_corr:
            best_val_corr = val_psth_corr
            epochs_without_improvement = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            break
    
    if best_state:
        torch.save({'epoch': best_epoch, 'model_state_dict': best_state, 'val_psth_corr': best_val_corr}, str(output_dir / 'model_best.pt'))
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump({'config': {'eye_mlp_hidden': eye_mlp_hidden}, 'best_val_corr': best_val_corr, 'history': history}, f, indent=2)
    
    return best_val_corr


def main():
    data_path = 'data/rnn_export_Newton_08_15_2025_SC.mat'
    base_dir = Path('results/eye_pathway_experiments')
    base_dir.mkdir(parents=True, exist_ok=True)
    
    experiments = [{'name': f'eye_mlp_{h}', 'eye_mlp_hidden': h} for h in [8, 16, 32]]
    results = []
    
    print("="*70 + "\nEXPERIMENT: Eye Position MLP\n" + "="*70)
    for exp in experiments:
        name = exp.pop('name')
        print(f"\n{name}")
        corr = train_eye_mlp(data_path=data_path, output_dir=str(base_dir / name), **exp)
        results.append({'name': name, 'best_val_corr': corr})
        exp['name'] = name
    
    print("\nRESULTS:")
    for r in sorted(results, key=lambda x: x['best_val_corr'], reverse=True):
        print(f"  {r['name']}: {r['best_val_corr']:.4f}")
    
    with open(base_dir / 'summary.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
