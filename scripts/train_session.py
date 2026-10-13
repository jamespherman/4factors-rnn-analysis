#!/usr/bin/env python3
"""
Train the E-I RNN on one session with correct neuron-to-unit mapping.

Differences from train_final_model.py / train_conditioned_loss.py:

1. Recorded neurons are fit by units of the matching type. Earlier scripts fit
   recorded neuron j with model unit j (model_rates[:, :, :n_recorded]); since
   recorded E and I neurons are interleaved in the export while model units
   0..n_exc-1 are excitatory, recorded interneurons were fit by E units and the
   model's I units were almost never constrained by data. Here recorded E
   neurons map to E units 0..n_E-1 and recorded I neurons to I units
   n_exc..n_exc+n_I-1 (saved as 'recorded_unit_indices' in config.json).

2. Condition-specific PSTH loss over the conditions the inputs distinguish:
   stimulus class (face, non-face, high-salience bullseye, low-salience
   bullseye) x target location = 16 conditions. (RNNDataset.get_condition_labels
   median-splits trial_salience, whose coding is 0 = image trial, 1 = high,
   2 = low, and conditions on reward, which the inputs carry no information about
   before reward delivery.)

3. Optional target-aligned PSTH loss ('cond_target' variant): the same
   condition-specific loss on a -100..+400 ms window around each trial's target
   onset. Target onset varies by ~0.5 s across trials, so fixation-aligned PSTHs
   blur the visual transient.

Model selection uses the validation condition-specific loss(es), not the grand
average PSTH correlation.

Usage:
    python scripts/train_session.py --session 08_15 --variant cond_target
    python scripts/train_session.py --session 08_15 --variant cond --test
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import create_model_from_data
from src.losses import compute_L_neuron_conditioned, compute_L_trial, compute_L_reg
from src.data_loader import load_mat_file, RNNDataset, validate_data, train_val_split
from src.analysis.pipeline_selectivity import get_target_onset_bins

CONFIG = {
    # Training (as train_conditioned_loss.py)
    'max_epochs': 2000,
    'patience': 300,
    'lr': 1e-3,
    'lr_scheduler_patience': 50,
    'lr_scheduler_factor': 0.5,
    'min_lr': 1e-5,
    'gradient_clip': 1.0,
    'seed': 42,
    'val_fraction': 0.2,

    # Model (as results/final_model)
    'tau': 50.0,
    'dt': 25.0,
    'noise_scale': 0.1,
    'spectral_radius': 0.9,
    'input_embed_dim': 56,
    'input_embed_type': 'attention',
    'attention_heads': 4,
    'learnable_h0': True,
    'h0_init': 0.1,

    # Loss
    'use_grad_balancing': True,
    'ltrial_scale': 0.5,
    'ltarget_scale': 1.0,
    'lambda_reg': 1e-4,
    'lambda_scale': 0.1,
    'lambda_var': 0.05,
    'min_trials_per_condition': 3,
    'target_window_ms': [-100.0, 400.0],

    'checkpoint_every': 250,
}

SESSION_FILES = {
    '08_13': 'data/rnn_export_Newton_08_13_2025_SC.mat',
    '08_14': 'data/rnn_export_Newton_08_14_2025_SC.mat',
    '08_15': 'data/rnn_export_Newton_08_15_2025_SC.mat',
}


def get_recorded_unit_indices(neuron_type: np.ndarray, n_exc: int) -> np.ndarray:
    """
    Model unit fit to each recorded neuron, preserving cell type.

    Args:
        neuron_type: [n_recorded] 1 = classic (E), 2 = putative interneuron (I)
        n_exc: Number of excitatory model units

    Returns:
        [n_recorded] model unit index for each recorded neuron
    """
    neuron_type = np.asarray(neuron_type).ravel()
    is_e = neuron_type == 1
    assert is_e.sum() <= n_exc, "More recorded E neurons than E units"

    indices = np.empty(len(neuron_type), dtype=np.int64)
    indices[is_e] = np.arange(is_e.sum())
    indices[~is_e] = n_exc + np.arange((~is_e).sum())
    return indices


def get_stimulus_location_conditions(data: dict) -> np.ndarray:
    """
    Condition label = stimulus class x target location (16 conditions).

    Stimulus class: 0 = face, 1 = non-face, 2 = high-salience bullseye,
    3 = low-salience bullseye (export coding: trial_identity 1/2/3,
    trial_salience 1 = high, 2 = low).

    Returns:
        [n_trials] int condition index 0-15
    """
    identity = np.asarray(data['trial_identity']).ravel()
    salience = np.asarray(data['trial_salience']).ravel()
    location = np.asarray(data['trial_location']).ravel().astype(int)

    stim_class = np.where(identity == 1, 0,
                 np.where(identity == 2, 1,
                 np.where(salience == 1, 2, 3)))
    assert set(np.unique(location)).issubset({1, 2, 3, 4}), "Unexpected location codes"
    return (stim_class * 4 + (location - 1)).astype(np.int64)


def get_target_window_index(
    onset_bins: np.ndarray, n_bins: int, bin_size_ms: float, window_ms: list
) -> tuple:
    """
    Gather indices for a target-aligned window.

    Returns:
        valid: [n_trials] bool, trials whose whole window lies inside the trial
        bin_index: [n_valid, n_window] time-bin indices for valid trials
    """
    start = int(round(window_ms[0] / bin_size_ms))
    end = int(round(window_ms[1] / bin_size_ms))
    offsets = np.arange(start, end)
    valid = (onset_bins >= 0) & (onset_bins + start >= 0) & (onset_bins + end <= n_bins)
    bin_index = onset_bins[valid, None] + offsets[None, :]
    return valid, bin_index


def target_aligned(rates: torch.Tensor, bin_index: torch.Tensor) -> torch.Tensor:
    """[n_valid, n_bins, n] -> [n_valid, n_window, n] using per-trial bin indices."""
    idx = bin_index.unsqueeze(-1).expand(-1, -1, rates.shape[-1])
    return torch.gather(rates, 1, idx)


def grand_psth_correlation(model_rates: np.ndarray, targets: np.ndarray) -> float:
    """Mean over neurons of the correlation between trial-averaged PSTHs."""
    m, t = model_rates.mean(axis=0), targets.mean(axis=0)
    r = [np.corrcoef(m[:, i], t[:, i])[0, 1] for i in range(t.shape[1])]
    return float(np.nanmean(r))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--session', required=True, choices=sorted(SESSION_FILES))
    parser.add_argument('--variant', required=True, choices=['cond', 'cond_target'])
    parser.add_argument('--seed', type=int, default=CONFIG['seed'])
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--max-epochs', type=int, default=None)
    parser.add_argument('--device', default=None, help='Default: cuda if available')
    parser.add_argument('--threads', type=int, default=None)
    parser.add_argument('--test', action='store_true', help='5-epoch smoke test')
    args = parser.parse_args()

    config = dict(CONFIG)
    config.update(session=args.session, variant=args.variant, seed=args.seed,
                  data_path=SESSION_FILES[args.session])
    if args.max_epochs:
        config['max_epochs'] = args.max_epochs
    if args.test:
        config.update(max_epochs=5, patience=3, checkpoint_every=2)
    out_dir = Path(args.output_dir or
                   f"results/retrained/{args.variant}/Newton_{args.session}_2025_SC_seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if args.threads:
        torch.set_num_threads(args.threads)

    # Data
    data = load_mat_file(config['data_path'])
    validate_data(data)
    dataset = RNNDataset(data)
    train_idx, val_idx = train_val_split(dataset, val_fraction=config['val_fraction'], seed=config['seed'])
    train_idx = np.sort(np.asarray(train_idx, dtype=np.int64))
    val_idx = np.sort(np.asarray(val_idx, dtype=np.int64))
    bin_size_ms = dataset.bin_size_ms
    assert bin_size_ms == config['dt'], "dt must equal the data bin size"

    all_trials = dataset.get_all_trials()
    inputs = all_trials['inputs'].to(device)
    targets = all_trials['targets'].to(device)
    mask = all_trials['mask'].to(device)
    conditions = torch.tensor(get_stimulus_location_conditions(data), device=device)

    onset_bins = get_target_onset_bins(data)
    tgt_valid, tgt_bins = get_target_window_index(
        onset_bins, dataset.n_bins, bin_size_ms, config['target_window_ms'])

    def split(idx):
        idx = np.asarray(idx, dtype=np.int64)
        sel_valid = tgt_valid[idx]
        # Rows of tgt_bins belonging to these trials
        row = np.cumsum(tgt_valid) - 1
        return {
            'idx': torch.tensor(idx, device=device),
            'tgt_trials': torch.tensor(idx[sel_valid], device=device),
            'tgt_bins': torch.tensor(tgt_bins[row[idx[sel_valid]]].astype(np.int64), device=device),
        }
    tr, va = split(train_idx), split(val_idx)

    # Model
    neuron_type = np.asarray(data['neuron_type']).ravel()
    n_e_rec, n_i_rec = int((neuron_type == 1).sum()), int((neuron_type == 2).sum())
    model = create_model_from_data(
        n_classic=n_e_rec, n_interneuron=n_i_rec, n_inputs=dataset.get_input_dim(),
        enforce_ratio=True, dt=config['dt'], tau=config['tau'],
        noise_scale=config['noise_scale'], spectral_radius=config['spectral_radius'],
        input_embed_dim=config['input_embed_dim'], input_embed_type=config['input_embed_type'],
        attention_heads=config['attention_heads'], learnable_h0=config['learnable_h0'],
        h0_init=config['h0_init'], device=device)
    assert model.verify_constraints()

    rec_units = get_recorded_unit_indices(neuron_type, model.n_exc)
    rec_units_t = torch.tensor(rec_units, device=device)
    assert np.all((rec_units < model.n_exc) == (neuron_type == 1)), "Unit types don't match neuron types"

    config.update(n_exc=model.n_exc, n_inh=model.n_inh, n_recorded_e=n_e_rec, n_recorded_i=n_i_rec,
                  n_inputs=dataset.get_input_dim(), arch_version='jan2026',
                  recorded_unit_indices=rec_units.tolist(),
                  train_idx=train_idx.tolist(), val_idx=val_idx.tolist(),
                  n_conditions=int(len(torch.unique(conditions))),
                  n_trials_target_window=int(tgt_valid.sum()), device=device)
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Session {args.session} | variant {args.variant} | seed {args.seed} | device {device}")
    print(f"  Recorded: {n_e_rec} E, {n_i_rec} I -> model {model.n_exc} E, {model.n_inh} I")
    print(f"  Trials: {len(train_idx)} train, {len(val_idx)} val; "
          f"{tgt_valid.sum()} with a full target-aligned window")

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config['lr_scheduler_factor'],
        patience=config['lr_scheduler_patience'], min_lr=config['min_lr'])

    use_target = args.variant == 'cond_target'
    cond_kw = dict(lambda_scale=config['lambda_scale'], lambda_var=config['lambda_var'],
                   min_trials_per_condition=config['min_trials_per_condition'])

    def losses(rates_rec, s):
        """Loss components on recorded units for trial subset s."""
        ii = s['idx']
        L_cond, _ = compute_L_neuron_conditioned(
            rates_rec[ii], targets[ii], conditions[ii], bin_size_ms, mask=mask[ii], **cond_kw)
        out = {'L_cond': L_cond}
        if use_target:
            jj = s['tgt_trials']
            L_tgt, _ = compute_L_neuron_conditioned(
                target_aligned(rates_rec[jj], s['tgt_bins']),
                target_aligned(targets[jj], s['tgt_bins']),
                conditions[jj], bin_size_ms, **cond_kw)
            out['L_target'] = L_tgt
        return out

    ema = {'L_cond': 1.0, 'L_target': 1.0, 'L_trial': 1.0}
    ema_decay = 0.99
    history = {k: [] for k in ['loss', 'L_cond', 'L_target', 'L_trial', 'val_L_cond', 'val_L_target',
                               'val_select', 'val_grand_corr', 'lr']}
    best = {'select': float('inf'), 'epoch': 0, 'state': None}
    t0 = time.time()

    for epoch in range(config['max_epochs']):
        model.train()
        optimizer.zero_grad()
        rates, _ = model(inputs[tr['idx']])
        # Place train rates in a full-size tensor so subset indices stay valid
        rates_rec = torch.zeros(len(dataset), rates.shape[1], len(rec_units), device=device)
        rates_rec = rates_rec.index_copy(0, tr['idx'], rates[:, :, rec_units_t])

        comp = losses(rates_rec, tr)
        comp['L_trial'] = compute_L_trial(
            rates_rec[tr['idx']], targets[tr['idx']], bin_size_ms,
            mask=mask[tr['idx']], sinkhorn_iters=20, sinkhorn_epsilon=0.1)
        L_reg = compute_L_reg(model, config['lambda_reg'])

        scales = {'L_cond': 1.0, 'L_target': config['ltarget_scale'], 'L_trial': config['ltrial_scale']}
        loss = L_reg
        for k, v in comp.items():
            if config['use_grad_balancing']:
                ema[k] = ema_decay * ema[k] + (1 - ema_decay) * v.item()
                loss = loss + scales[k] * v / (ema[k] + 1e-8)
            else:
                loss = loss + scales[k] * v
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['gradient_clip'])
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            rates_v, _ = model(inputs[va['idx']])
            rates_rec_v = torch.zeros(len(dataset), rates_v.shape[1], len(rec_units), device=device)
            rates_rec_v = rates_rec_v.index_copy(0, va['idx'], rates_v[:, :, rec_units_t])
            vcomp = losses(rates_rec_v, va)
            select = sum(v.item() for v in vcomp.values())
            grand = grand_psth_correlation(rates_rec_v[va['idx']].cpu().numpy(),
                                           targets[va['idx']].cpu().numpy())
        scheduler.step(select)

        history['loss'].append(loss.item())
        for k in ['L_cond', 'L_target', 'L_trial']:
            history[k].append(comp[k].item() if k in comp else None)
        history['val_L_cond'].append(vcomp['L_cond'].item())
        history['val_L_target'].append(vcomp['L_target'].item() if 'L_target' in vcomp else None)
        history['val_select'].append(select)
        history['val_grand_corr'].append(grand)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        if select < best['select']:
            best = {'select': select, 'epoch': epoch + 1,
                    'state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1:4d} | loss {loss.item():.4f} | val_select {select:.4f} "
                  f"| val_grand_corr {grand:.4f} | best {best['select']:.4f}@{best['epoch']} "
                  f"| lr {optimizer.param_groups[0]['lr']:.1e} | {time.time() - t0:.0f}s", flush=True)
        if (epoch + 1) % config['checkpoint_every'] == 0:
            torch.save(model.state_dict(), out_dir / f'model_epoch{epoch + 1}.pt')
        if epoch + 1 - best['epoch'] >= config['patience']:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    torch.save(best['state'], out_dir / 'model_best.pt')
    torch.save(model.state_dict(), out_dir / 'model_final.pt')
    summary = {'best_epoch': best['epoch'], 'best_val_select': best['select'],
               'val_grand_corr_at_best': history['val_grand_corr'][best['epoch'] - 1],
               'epochs_trained': epoch + 1, 'training_time_s': time.time() - t0}
    with open(out_dir / 'history.json', 'w') as f:
        json.dump(history, f)
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Done: {json.dumps(summary)}")


if __name__ == '__main__':
    main()
