"""Experiment: Data Augmentation"""
import json, torch, numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model import create_model_from_data
from src.losses import compute_L_neuron, compute_L_reg
from src.data_loader import load_session, train_val_split

def augment_inputs(inputs, eye_noise_scale=0.1):
    aug = inputs.clone(); aug[..., 7:9] += torch.randn_like(aug[..., 7:9]) * eye_noise_scale
    return aug

def augment_targets(targets, noise_scale=0.05):
    return torch.clamp(targets + torch.randn_like(targets) * noise_scale * (targets.mean() + 1e-8), min=0)

def compute_psth_correlation(model, data, device):
    model.eval()
    with torch.no_grad():
        inputs, targets = data['inputs'].to(device), data['targets'].to(device)
        model_rates, _ = model(inputs)
        n = targets.shape[2]
        mp, tp = model_rates[:,:,:n].mean(0).cpu().numpy(), targets.mean(0).cpu().numpy()
        return np.nanmean([np.corrcoef(mp[:,i], tp[:,i])[0,1] for i in range(n)])

def train_with_augmentation(data_path, output_dir, eye_noise_scale=0.0, target_noise_scale=0.0, max_epochs=500, patience=100, device='cpu', seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_session(data_path, validate=False)
    train_idx, val_idx = train_val_split(dataset, val_fraction=0.2, seed=seed)
    all_data = dataset.get_all_trials()
    train_data = {k: all_data[k][train_idx] for k in ['inputs', 'targets', 'mask']}
    val_data = {k: all_data[k][val_idx] for k in ['inputs', 'targets', 'mask']}
    neuron_info = dataset.get_neuron_info()
    model = create_model_from_data(n_classic=neuron_info['n_exc'], n_interneuron=neuron_info['n_inh'], n_inputs=dataset.get_input_dim(), dt=float(dataset.bin_size_ms), device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=50, min_lr=1e-5)
    history = {'train_loss': [], 'val_psth_corr': []}
    best_val_corr, best_state, best_epoch, no_improve = float('-inf'), None, 0, 0
    for epoch in tqdm(range(max_epochs), desc=f"eye={eye_noise_scale},tgt={target_noise_scale}"):
        model.train()
        inputs, targets, mask = train_data['inputs'].to(device), train_data['targets'].to(device), train_data['mask'].to(device)
        if eye_noise_scale > 0: inputs = augment_inputs(inputs, eye_noise_scale)
        if target_noise_scale > 0: targets = augment_targets(targets, target_noise_scale)
        model_rates, _ = model(inputs)
        loss = compute_L_neuron(model_rates[:,:,:train_data['targets'].shape[2]], targets, bin_size_ms=dataset.bin_size_ms, mask=mask) + compute_L_reg(model, lambda_l2=1e-4)
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        val_corr = compute_psth_correlation(model, val_data, device)
        scheduler.step(val_corr)
        history['train_loss'].append(loss.item()); history['val_psth_corr'].append(val_corr)
        if val_corr > best_val_corr: best_val_corr, no_improve, best_state, best_epoch = val_corr, 0, {k: v.cpu().clone() for k,v in model.state_dict().items()}, epoch
        else: no_improve += 1
        if no_improve >= patience: break
    if best_state: torch.save({'epoch': best_epoch, 'model_state_dict': best_state, 'val_psth_corr': best_val_corr}, str(output_dir / 'model_best.pt'))
    with open(output_dir / 'results.json', 'w') as f: json.dump({'config': {'eye_noise_scale': eye_noise_scale, 'target_noise_scale': target_noise_scale}, 'best_val_corr': best_val_corr, 'history': history}, f, indent=2)
    return best_val_corr

def main():
    data_path = 'data/rnn_export_Newton_08_15_2025_SC.mat'
    base_dir = Path('results/augmentation_experiments'); base_dir.mkdir(parents=True, exist_ok=True)
    experiments = [{'name': 'no_aug', 'eye_noise_scale': 0.0}, {'name': 'eye_0.1', 'eye_noise_scale': 0.1}, {'name': 'eye_0.2', 'eye_noise_scale': 0.2}, {'name': 'tgt_0.05', 'target_noise_scale': 0.05}, {'name': 'both', 'eye_noise_scale': 0.1, 'target_noise_scale': 0.05}]
    results = []
    print("="*70 + "\nEXPERIMENT: Data Augmentation\n" + "="*70)
    for exp in experiments:
        name = exp.pop('name'); print(f"\n{name}")
        corr = train_with_augmentation(data_path=data_path, output_dir=str(base_dir / name), **exp)
        results.append({'name': name, 'best_val_corr': corr}); exp['name'] = name
    print("\nRESULTS:")
    for r in sorted(results, key=lambda x: x['best_val_corr'], reverse=True): print(f"  {r['name']}: {r['best_val_corr']:.4f}")
    with open(base_dir / 'summary.json', 'w') as f: json.dump(results, f, indent=2)

if __name__ == "__main__": main()
