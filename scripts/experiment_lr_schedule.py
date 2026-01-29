"""Experiment: Learning Rate Schedules"""
import json, math, torch, numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model import create_model_from_data
from src.losses import compute_L_neuron, compute_L_reg
from src.data_loader import load_session, train_val_split

def compute_psth_correlation(model, data, device):
    model.eval()
    with torch.no_grad():
        inputs, targets = data['inputs'].to(device), data['targets'].to(device)
        model_rates, _ = model(inputs)
        n = targets.shape[2]
        mp, tp = model_rates[:,:,:n].mean(0).cpu().numpy(), targets.mean(0).cpu().numpy()
        return np.nanmean([np.corrcoef(mp[:,i], tp[:,i])[0,1] for i in range(n)])

def get_warmup_cosine_lr(epoch, warmup, total, base_lr, min_lr):
    if epoch < warmup: return base_lr * epoch / warmup
    progress = (epoch - warmup) / (total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))

def train_with_schedule(data_path, output_dir, schedule_type='plateau', warmup_epochs=0, T_0=100, max_epochs=500, patience=100, device='cpu', seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_session(data_path, validate=False)
    train_idx, val_idx = train_val_split(dataset, val_fraction=0.2, seed=seed)
    all_data = dataset.get_all_trials()
    train_data = {k: all_data[k][train_idx] for k in ['inputs', 'targets', 'mask']}
    val_data = {k: all_data[k][val_idx] for k in ['inputs', 'targets', 'mask']}
    neuron_info = dataset.get_neuron_info()
    model = create_model_from_data(n_classic=neuron_info['n_exc'], n_interneuron=neuron_info['n_inh'], n_inputs=dataset.get_input_dim(), dt=float(dataset.bin_size_ms), device=device)
    base_lr, min_lr = 1e-3, 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=50, min_lr=min_lr) if schedule_type == 'plateau' else (torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=2, eta_min=min_lr) if schedule_type == 'cosine' else None)
    history = {'train_loss': [], 'val_psth_corr': [], 'lr': []}
    best_val_corr, best_state, best_epoch, no_improve = float('-inf'), None, 0, 0
    for epoch in tqdm(range(max_epochs), desc=f"sched={schedule_type}"):
        if schedule_type == 'warmup_cosine':
            for pg in optimizer.param_groups: pg['lr'] = get_warmup_cosine_lr(epoch, warmup_epochs, max_epochs, base_lr, min_lr)
        model.train()
        inputs, targets, mask = [train_data[k].to(device) for k in ['inputs', 'targets', 'mask']]
        model_rates, _ = model(inputs)
        loss = compute_L_neuron(model_rates[:,:,:targets.shape[2]], targets, bin_size_ms=dataset.bin_size_ms, mask=mask) + compute_L_reg(model, lambda_l2=1e-4)
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        val_corr = compute_psth_correlation(model, val_data, device)
        if schedule_type == 'plateau': scheduler.step(val_corr)
        elif schedule_type == 'cosine': scheduler.step()
        history['train_loss'].append(loss.item()); history['val_psth_corr'].append(val_corr); history['lr'].append(optimizer.param_groups[0]['lr'])
        if val_corr > best_val_corr: best_val_corr, no_improve, best_state, best_epoch = val_corr, 0, {k: v.cpu().clone() for k,v in model.state_dict().items()}, epoch
        else: no_improve += 1
        if no_improve >= patience: break
    if best_state: torch.save({'epoch': best_epoch, 'model_state_dict': best_state, 'val_psth_corr': best_val_corr}, str(output_dir / 'model_best.pt'))
    with open(output_dir / 'results.json', 'w') as f: json.dump({'config': {'schedule_type': schedule_type}, 'best_val_corr': best_val_corr, 'history': history}, f, indent=2)
    return best_val_corr

def main():
    data_path = 'data/rnn_export_Newton_08_15_2025_SC.mat'
    base_dir = Path('results/lr_schedule_experiments'); base_dir.mkdir(parents=True, exist_ok=True)
    experiments = [{'name': 'plateau_baseline', 'schedule_type': 'plateau'}, {'name': 'cosine_T100', 'schedule_type': 'cosine', 'T_0': 100}, {'name': 'cosine_T50', 'schedule_type': 'cosine', 'T_0': 50}, {'name': 'warmup_cosine_50', 'schedule_type': 'warmup_cosine', 'warmup_epochs': 50}, {'name': 'warmup_cosine_100', 'schedule_type': 'warmup_cosine', 'warmup_epochs': 100}]
    results = []
    print("="*70 + "\nEXPERIMENT: LR Schedules\n" + "="*70)
    for exp in experiments:
        name = exp.pop('name'); print(f"\n{name}")
        corr = train_with_schedule(data_path=data_path, output_dir=str(base_dir / name), **exp)
        results.append({'name': name, 'best_val_corr': corr}); exp['name'] = name
    print("\nRESULTS:")
    for r in sorted(results, key=lambda x: x['best_val_corr'], reverse=True): print(f"  {r['name']}: {r['best_val_corr']:.4f}")
    with open(base_dir / 'summary.json', 'w') as f: json.dump(results, f, indent=2)

if __name__ == "__main__": main()
