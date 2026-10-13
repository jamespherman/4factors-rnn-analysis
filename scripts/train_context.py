#!/usr/bin/env python3
"""
Train E-I RNNs with task-context inputs (specs/CONTEXT_INPUTS_PLAN.md, Stages B-C).

Builds on scripts/train_session.py (corrected E/I mapping, 'cond_target' loss) and adds:

- Context inputs (--context):
    none:       no context channels (negative control)
    shared:     block state (reward map, block; coded +/-1, on for the whole trial)
                through one weight per channel per population (E, I). Unit-specific
                context effects must then come from the recurrent circuit. (primary)
    free:       same channels, free per-unit weights (upper bound; lets each unit
                read "my location is valuable" directly)
    trialvalue: current target's reward and probability (+/-1) from target onset
                (positive control: hands the model the conjunction)
  --context-targets all | e | i routes context to all units, E only or I only.
- Slow drift inputs in every variant: Legendre polynomials 1-3 of trial index, free
  per-unit weights, so context weights don't absorb recording drift.
- I->E constraint ladder (--ie-mode free | spatial | global | frozen, see EIRNN).
- Optional hidden interneurons (--hidden-i), keeping the 4:1 E:I ratio.
- Context-conditioned PSTH losses: location x context state (16 cells), on the
  fixation-aligned trial and the target-aligned window, in every variant.
- Trials split 70/15/15 (train / selection / test), stratified by location x
  context. Checkpoint selection uses the selection split only, with each validation
  loss normalized by its value at epoch 1.
- --data overrides the export file (for synthetic data in the recovery test).

Usage:
    python scripts/train_context.py --session 08_15 --context shared --ie-mode free --seed 1
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src.model import EIRNN
from src.losses import compute_L_neuron_conditioned, compute_L_trial, compute_L_reg
from src.data_loader import load_mat_file, RNNDataset, validate_data
from src.analysis.pipeline_selectivity import (
    get_target_onset_bins, compute_window_rates, compute_preferred_locations
)
from src.analysis.task_context import get_task_context
from train_session import (
    CONFIG as BASE_CONFIG, SESSION_FILES, get_recorded_unit_indices,
    get_stimulus_location_conditions, get_target_window_index, target_aligned,
    grand_psth_correlation
)

CONFIG = dict(BASE_CONFIG)
CONFIG.update({
    'n_drift': 3,
    'test_fraction': 0.15,
    'select_fraction': 0.15,
    'lctx_scale': 1.0,
    'checkpoint_every': 50,
})


def get_context_channels(data: dict, mode: str, onset_bins: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Context input channels [n_trials, n_bins, n_channels] (empty for 'none').

    Block state: reward map (+1 = locations 1, 2 rewarded) and block (+1 = location 1
    frequent), constant over the trial. Trial value: current target's reward and
    probability (+1 = high), from the target onset bin onwards.
    """
    n_trials = len(onset_bins)
    if mode == 'none':
        return np.zeros((n_trials, n_bins, 0), dtype=np.float32)
    if mode in ('shared', 'free'):
        ctx = get_task_context(data)
        block_code = np.where(ctx['block'] == 1, 1.0, np.where(ctx['block'] == 2, -1.0, 0.0))  # 0 if unknown
        values = np.stack([2.0 * ctx['reward_map'] - 1, block_code], axis=1)
        return np.repeat(values[:, None, :], n_bins, axis=1).astype(np.float32)
    assert mode == 'trialvalue', f"Unknown context mode: {mode}"
    reward = 2.0 * np.asarray(data['trial_reward']).ravel() - 1
    probability = 2.0 * np.asarray(data['trial_probability']).ravel() - 1
    channels = np.zeros((n_trials, n_bins, 2), dtype=np.float32)
    for t in range(n_trials):
        if onset_bins[t] >= 0:
            channels[t, onset_bins[t]:] = (reward[t], probability[t])
    return channels


def get_drift_channels(n_trials: int, n_bins: int, n_drift: int) -> np.ndarray:
    """Legendre polynomials 1..n_drift of normalized trial index, constant within a trial."""
    x = np.linspace(-1, 1, n_trials)
    basis = np.stack([np.polynomial.legendre.Legendre.basis(k)(x) for k in range(1, n_drift + 1)], axis=1)
    return np.repeat(basis[:, None, :], n_bins, axis=1).astype(np.float32)


def get_context_conditions(data: dict) -> np.ndarray:
    """Location x context state (reward map x block) = 16 conditions; -1 if block unknown."""
    ctx = get_task_context(data)
    location = np.asarray(data['trial_location']).ravel().astype(int)
    cond = (location - 1) * 4 + ctx['reward_map'] * 2 + (ctx['block'] == 1)
    cond[np.isnan(ctx['block'])] = -1
    return cond.astype(np.int64)


def three_way_split(strata: np.ndarray, test_fraction: float, select_fraction: float, seed: int):
    """Stratified train / selection / test split: each stratum is shuffled and divided."""
    rng = np.random.default_rng(seed)
    train, select, test = [], [], []
    for value in np.unique(strata):
        idx = rng.permutation(np.flatnonzero(strata == value))
        n_test = int(round(test_fraction * len(idx)))
        n_select = int(round(select_fraction * len(idx)))
        test.extend(idx[:n_test])
        select.extend(idx[n_test:n_test + n_select])
        train.extend(idx[n_test + n_select:])
    return np.sort(train), np.sort(select), np.sort(test)


def get_unit_locations(data: dict, rec_units: np.ndarray, n_total: int) -> np.ndarray:
    """Preferred location per unit: recorded neurons' visual maximum; hidden units cycle 1-4."""
    onset = get_target_onset_bins(data)
    vis = compute_window_rates(np.transpose(data['firing_rates'], (2, 1, 0)), onset)
    pref = compute_preferred_locations(vis, np.asarray(data['trial_location']).ravel())['for_factors']
    locations = (np.arange(n_total) % 4) + 1
    locations[rec_units] = pref
    return locations


def main():
    parser = argparse.ArgumentParser(description='Train E-I RNN with task-context inputs')
    parser.add_argument('--session', required=True, choices=sorted(SESSION_FILES))
    parser.add_argument('--context', required=True, choices=['none', 'shared', 'free', 'trialvalue'])
    parser.add_argument('--context-targets', default='all', choices=['all', 'e', 'i'])
    parser.add_argument('--ie-mode', default='free', choices=['free', 'spatial', 'global', 'frozen'])
    parser.add_argument('--frozen-from', default=None,
                        help="Model dir whose I->E block (permuted) is frozen for --ie-mode frozen")
    parser.add_argument('--hidden-i', type=int, default=0, help='Extra unrecorded I units')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--data', default=None, help='Export file (default: the session export)')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-epochs', type=int, default=None)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--test', action='store_true', help='5-epoch smoke test')
    args = parser.parse_args()

    config = dict(CONFIG)
    config.update(session=args.session, context=args.context, context_targets=args.context_targets,
                  ie_mode=args.ie_mode, frozen_from=args.frozen_from, hidden_i=args.hidden_i,
                  seed=args.seed, data_path=args.data or SESSION_FILES[args.session])
    if args.max_epochs:
        config['max_epochs'] = args.max_epochs
    if args.test:
        config.update(max_epochs=5, patience=3, checkpoint_every=2)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / 'summary.json').exists():
        print(f"Already finished: {out_dir}")
        return

    torch.set_num_threads(args.threads)
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    device = args.device

    # Data
    data = load_mat_file(config['data_path'])
    validate_data(data)
    dataset = RNNDataset(data)
    n_trials, n_bins = len(dataset), dataset.n_bins
    bin_size_ms = dataset.bin_size_ms
    assert bin_size_ms == config['dt'], "dt must equal the data bin size"

    onset_bins = get_target_onset_bins(data)
    ctx_channels = get_context_channels(data, args.context, onset_bins, n_bins)
    drift_channels = get_drift_channels(n_trials, n_bins, config['n_drift'])
    all_trials = dataset.get_all_trials()
    n_task_inputs = all_trials['inputs'].shape[-1]
    inputs = torch.cat([all_trials['inputs'], torch.from_numpy(ctx_channels),
                        torch.from_numpy(drift_channels)], dim=-1).to(device)
    targets = all_trials['targets'].to(device)
    mask = all_trials['mask'].to(device)

    stim_cond = get_stimulus_location_conditions(data)
    ctx_cond = get_context_conditions(data)
    conditions = {'stim': torch.tensor(stim_cond, device=device),
                  'ctx': torch.tensor(ctx_cond, device=device)}

    train_idx, select_idx, test_idx = three_way_split(
        ctx_cond, config['test_fraction'], config['select_fraction'], config['seed'])

    tgt_valid, tgt_bins = get_target_window_index(onset_bins, n_bins, bin_size_ms, config['target_window_ms'])
    row = np.cumsum(tgt_valid) - 1

    def split(idx):
        idx = np.asarray(idx, dtype=np.int64)
        ctx_ok = ctx_cond[idx] >= 0
        tv = tgt_valid[idx]
        tv_ctx = tv & ctx_ok
        return {
            'idx': torch.tensor(idx, device=device),
            'ctx_idx': torch.tensor(idx[ctx_ok], device=device),
            'tgt_trials': torch.tensor(idx[tv], device=device),
            'tgt_bins': torch.tensor(tgt_bins[row[idx[tv]]].astype(np.int64), device=device),
            'tgt_ctx_trials': torch.tensor(idx[tv_ctx], device=device),
            'tgt_ctx_bins': torch.tensor(tgt_bins[row[idx[tv_ctx]]].astype(np.int64), device=device),
        }
    tr, se = split(train_idx), split(select_idx)

    # Model
    neuron_type = np.asarray(data['neuron_type']).ravel()
    n_e_rec, n_i_rec = int((neuron_type == 1).sum()), int((neuron_type == 2).sum())
    n_inh = n_i_rec + args.hidden_i
    n_exc = max(n_e_rec, int(np.ceil(4 * n_inh)))
    rec_units = get_recorded_unit_indices(neuron_type, n_exc)
    unit_locations = get_unit_locations(data, rec_units, n_exc + n_inh)
    n_context = ctx_channels.shape[-1]

    model = EIRNN(
        n_exc=n_exc, n_inh=n_inh, n_inputs=n_task_inputs, dt=config['dt'], tau=config['tau'],
        noise_scale=config['noise_scale'], spectral_radius=config['spectral_radius'],
        input_embed_dim=config['input_embed_dim'], input_embed_type=config['input_embed_type'],
        attention_heads=config['attention_heads'], learnable_h0=config['learnable_h0'],
        h0_init=config['h0_init'],
        n_context=n_context, context_mode='free' if args.context in ('free', 'trialvalue') else 'shared',
        context_targets=args.context_targets, n_drift=config['n_drift'],
        ie_mode=args.ie_mode, unit_locations=unit_locations, device=device)
    if args.ie_mode == 'frozen':
        assert args.frozen_from, "--ie-mode frozen needs --frozen-from"
        donor = torch.load(Path(args.frozen_from) / 'model_best.pt', map_location=device, weights_only=False)
        W_EI = torch.abs(donor['W_rec_raw'][:n_exc, n_exc:])
        assert W_EI.shape == (n_exc, n_inh), "Donor model has a different size"
        gen = torch.Generator().manual_seed(config['seed'])
        for i in range(n_inh):  # permute each I unit's targets (keeps its outgoing distribution)
            W_EI[:, i] = W_EI[torch.randperm(n_exc, generator=gen), i]
        model.set_frozen_ie(W_EI)
    assert model.verify_constraints()
    rec_units_t = torch.tensor(rec_units, device=device)

    config.update(n_exc=n_exc, n_inh=n_inh, n_recorded_e=n_e_rec, n_recorded_i=n_i_rec,
                  n_inputs=n_task_inputs, n_context=n_context, arch_version='jan2026',
                  context_mode=model.context_mode, recorded_unit_indices=rec_units.tolist(),
                  unit_locations=unit_locations.tolist(),
                  train_idx=train_idx.tolist(), select_idx=select_idx.tolist(), test_idx=test_idx.tolist(),
                  device=device)
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Session {args.session} | context {args.context}->{args.context_targets} | ie {args.ie_mode} "
          f"| hidden I {args.hidden_i} | seed {args.seed}")
    print(f"  Recorded: {n_e_rec} E, {n_i_rec} I -> model {n_exc} E, {n_inh} I; "
          f"trials {len(train_idx)} train / {len(select_idx)} select / {len(test_idx)} test", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config['lr_scheduler_factor'],
        patience=config['lr_scheduler_patience'], min_lr=config['min_lr'])
    cond_kw = dict(lambda_scale=config['lambda_scale'], lambda_var=config['lambda_var'],
                   min_trials_per_condition=config['min_trials_per_condition'])

    def losses(rates_rec, s):
        """Conditioned PSTH losses on recorded units for trial subset s."""
        out = {}
        for name, cond, ii in [('L_cond', 'stim', s['idx']), ('L_ctx', 'ctx', s['ctx_idx'])]:
            out[name], _ = compute_L_neuron_conditioned(
                rates_rec[ii], targets[ii], conditions[cond][ii], bin_size_ms, mask=mask[ii], **cond_kw)
        for name, cond, jj, bins in [('L_target', 'stim', s['tgt_trials'], s['tgt_bins']),
                                     ('L_ctx_target', 'ctx', s['tgt_ctx_trials'], s['tgt_ctx_bins'])]:
            out[name], _ = compute_L_neuron_conditioned(
                target_aligned(rates_rec[jj], bins), target_aligned(targets[jj], bins),
                conditions[cond][jj], bin_size_ms, **cond_kw)
        return out

    scales = {'L_cond': 1.0, 'L_target': config['ltarget_scale'], 'L_ctx': config['lctx_scale'],
              'L_ctx_target': config['lctx_scale'] * config['ltarget_scale'], 'L_trial': config['ltrial_scale']}
    ema = {k: 1.0 for k in scales}
    ema_decay = 0.99
    history = {k: [] for k in ['loss', 'val_select', 'val_grand_corr', 'lr'] + list(scales)
               + [f'val_{k}' for k in scales if k != 'L_trial']}
    best = {'select': float('inf'), 'epoch': 0, 'state': None}
    val_ref = None
    start_epoch, elapsed_before = 0, 0.0
    ckpt_path = out_dir / 'checkpoint.pt'
    if ckpt_path.exists():  # resume after preemption
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        best, val_ref, ema, history = ckpt['best'], ckpt['val_ref'], ckpt['ema'], ckpt['history']
        start_epoch, elapsed_before = ckpt['epoch'], ckpt['elapsed']
        torch.manual_seed(config['seed'] + start_epoch)
        print(f"  Resumed from epoch {start_epoch}", flush=True)
    t0 = time.time() - elapsed_before

    def full_size(rates, idx):
        out = torch.zeros(n_trials, rates.shape[1], len(rec_units), device=device)
        return out.index_copy(0, idx, rates[:, :, rec_units_t])

    for epoch in range(start_epoch, config['max_epochs']):
        model.train()
        optimizer.zero_grad()
        rates, _ = model(inputs[tr['idx']])
        rates_rec = full_size(rates, tr['idx'])
        comp = losses(rates_rec, tr)
        comp['L_trial'] = compute_L_trial(
            rates_rec[tr['idx']], targets[tr['idx']], bin_size_ms,
            mask=mask[tr['idx']], sinkhorn_iters=20, sinkhorn_epsilon=0.1)
        loss = compute_L_reg(model, config['lambda_reg'])
        for k, v in comp.items():
            ema[k] = ema_decay * ema[k] + (1 - ema_decay) * v.item()
            loss = loss + scales[k] * v / (ema[k] + 1e-8)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['gradient_clip'])
        optimizer.step()

        model.eval()
        with torch.no_grad():
            rates_v, _ = model(inputs[se['idx']])
            rates_rec_v = full_size(rates_v, se['idx'])
            vcomp = {k: v.item() for k, v in losses(rates_rec_v, se).items()}
            if val_ref is None:
                val_ref = dict(vcomp)
            select = sum(vcomp[k] / (val_ref[k] + 1e-8) for k in vcomp)
            grand = grand_psth_correlation(rates_rec_v[se['idx']].cpu().numpy(),
                                           targets[se['idx']].cpu().numpy())
        scheduler.step(select)

        history['loss'].append(loss.item())
        for k in scales:
            history[k].append(comp[k].item())
            if k != 'L_trial':
                history[f'val_{k}'].append(vcomp[k])
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
            tmp = out_dir / 'checkpoint.tmp'
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'best': best, 'val_ref': val_ref,
                        'ema': ema, 'history': history, 'epoch': epoch + 1,
                        'elapsed': time.time() - t0}, tmp)
            tmp.replace(ckpt_path)
        if epoch + 1 - best['epoch'] >= config['patience']:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    torch.save(best['state'], out_dir / 'model_best.pt')
    summary = {'best_epoch': best['epoch'], 'best_val_select': best['select'],
               'val_grand_corr_at_best': history['val_grand_corr'][best['epoch'] - 1],
               'epochs_trained': epoch + 1, 'training_time_s': time.time() - t0}
    with open(out_dir / 'history.json', 'w') as f:
        json.dump(history, f)
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    ckpt_path.unlink(missing_ok=True)
    print(f"Done: {json.dumps(summary)}")


if __name__ == '__main__':
    main()
