#!/usr/bin/env python3
"""
Evaluate context models from scripts/train_context.py (specs/CONTEXT_INPUTS_PLAN.md).

For each trained model (config dir with model_best.pt):

1. Held-out fit: Poisson log-likelihood of the recorded spike counts on test trials
   under the model's mean rate (averaged over noise draws), per recorded neuron.
2. Context effects, computed on model rates with the same statistics as the
   recordings (scripts/analyze_context_effects.py), all trials:
   - pre-target reward-context and block effects (switch-triggered, K = 30)
   - visual probability component (sign-reversal test, (d1 - d3) / 2)
   - spatial selectivity (visual response, preferred minus opposite location)
   Reported as population means (model vs recorded) and neuron-wise correlation.
3. Current decomposition for recorded E units: the difference in mean input current
   between context states (pre-target window, preferred location rewarded vs not),
   split into direct context input, task input, drift input, E->E and I->E.

Usage:
    python scripts/evaluate_context_models.py --root results/context_pilot
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src.model import EIRNN
from src.data_loader import load_mat_file, RNNDataset
from src.analysis.pipeline_selectivity import get_target_onset_bins, compute_window_rates, OPPOSITE_LOCATION
from src.analysis.task_context import location_rewarded
from train_context import get_context_channels, get_drift_channels
import analyze_context_effects as A

warnings.filterwarnings('ignore', category=RuntimeWarning)

N_DRAWS = 5
SWITCH_K = 30


def load_context_model(model_dir: Path, device: str = 'cpu'):
    """Rebuild an EIRNN from a train_context.py (or train_session.py) config and load model_best.pt."""
    with open(model_dir / 'config.json') as f:
        config = json.load(f)
    # train_session.py configs: no context, drift or test split
    for key, value in [('context', 'none'), ('n_context', 0), ('context_mode', 'shared'),
                       ('context_targets', 'all'), ('n_drift', 0), ('ie_mode', 'free'),
                       ('hidden_i', 0), ('test_idx', [])]:
        config.setdefault(key, value)
    model = EIRNN(
        n_exc=config['n_exc'], n_inh=config['n_inh'], n_inputs=config['n_inputs'],
        dt=config['dt'], tau=config['tau'], noise_scale=config['noise_scale'],
        spectral_radius=config['spectral_radius'], input_embed_dim=config['input_embed_dim'],
        input_embed_type=config['input_embed_type'], attention_heads=config['attention_heads'],
        learnable_h0=config['learnable_h0'], h0_init=config['h0_init'],
        n_context=config['n_context'], context_mode=config['context_mode'],
        context_targets=config['context_targets'], n_drift=config['n_drift'],
        ie_mode=config['ie_mode'],
        unit_locations=np.array(config['unit_locations']) if 'unit_locations' in config else None, device=device)
    model.load_state_dict(torch.load(model_dir / 'model_best.pt', map_location=device, weights_only=False))
    model.eval()
    return model, config


def build_inputs(data: dict, config: dict) -> torch.Tensor:
    """Task + context + drift input tensor, as in training."""
    dataset = RNNDataset(data)
    onset = get_target_onset_bins(data)
    ctx = get_context_channels(data, config['context'], onset, dataset.n_bins)
    drift = get_drift_channels(len(dataset), dataset.n_bins, config['n_drift'])
    return torch.cat([dataset.get_all_trials()['inputs'], torch.from_numpy(ctx), torch.from_numpy(drift)], dim=-1)


def run_with_currents(model: EIRNN, inputs: torch.Tensor, seed: int):
    """
    Forward pass returning output rates and per-step input current components.

    Returns:
        rates: [trials, bins, n_total] output rates (sp/s)
        currents: dict of [trials, bins, n_total] arrays: 'EE' (from E units),
            'IE' (from I units), 'task', 'context', 'drift'
    """
    torch.manual_seed(seed)
    with torch.no_grad():
        rates, _, states = model(inputs, return_states=True)
        n_exc = model.n_exc
        r_raw = torch.nn.functional.softplus(states)
        # State at step t was driven by rates at step t-1 (initial state drives step 0)
        h0 = model.h0.unsqueeze(0).expand(inputs.shape[0], -1) if model.h0 is not None else torch.zeros_like(states[:, 0])
        r_prev = torch.cat([torch.nn.functional.softplus(h0).unsqueeze(1), r_raw[:, :-1]], dim=1)
        W = model.W_rec
        currents = {'EE': r_prev[:, :, :n_exc] @ W[:, :n_exc].T,
                    'IE': r_prev[:, :, n_exc:] @ W[:, n_exc:].T}
        task = inputs[:, :, :model.n_inputs]
        if model.input_embed is not None:
            task = model.input_embed(task)
        currents['task'] = task @ model.W_in.T
        extra = inputs[:, :, model.n_inputs:]
        if model.n_context > 0:
            currents['context'] = extra[:, :, :model.n_context] @ model.W_ctx.T
        if model.n_drift > 0:
            currents['drift'] = extra[:, :, model.n_context:] @ model.W_drift.T
    return rates.numpy(), {k: v.numpy() for k, v in currents.items()}


def poisson_ll(rates: np.ndarray, counts: np.ndarray, mask: np.ndarray, bin_s: float) -> np.ndarray:
    """Mean Poisson log-likelihood per valid bin, per neuron (constant log(k!) dropped)."""
    mu = np.clip(rates, 1e-3, None) * bin_s
    ll = counts * np.log(mu) - mu
    valid = mask.reshape(-1) > 0
    return np.nanmean(ll.reshape(-1, ll.shape[-1])[valid], axis=0)


def effects(sess: dict, min_trials: int = 5) -> dict:
    """Per-neuron context and spatial effects for a session dict (recorded or model rates)."""
    rows = A.switch_contrasts(sess, SWITCH_K)
    out = {}
    for ctx in ('reward', 'block'):
        per_neuron = {}
        for r in rows:
            if r['context'] == ctx and r['epoch'] == 'pre':
                per_neuron.setdefault(r['neuron'], []).append(r['diff'])
        v = np.full(sess['vis'].shape[1], np.nan)
        for i, d in per_neuron.items():
            v[i] = np.mean(d)
        out[f'{ctx}_pre'] = v
    sr = np.array(A.probability_sign_reversal(sess, min_trials), dtype=float)
    out['prob_vis'] = (sr[:, 0] - sr[:, 1]) / 2
    loc, vis = sess['location'], sess['vis']
    pref = sess['preferred']['for_factors']
    out['spatial'] = np.array([np.nanmean(vis[loc == p, i]) - np.nanmean(vis[loc == OPPOSITE_LOCATION[p], i])
                               for i, p in enumerate(pref)])
    return out


def evaluate(model_dir: Path, sessions: dict) -> dict:
    """Evaluate one trained model."""
    model, config = load_context_model(model_dir)
    sess = sessions[config['session']]
    data = load_mat_file(config['data_path']) if 'synthetic' in config['data_path'] else sess['data']
    inputs = build_inputs(data, config)
    rec_units = np.array(config['recorded_unit_indices'])
    n_exc = config['n_exc']

    rates, currents = None, None
    for seed in range(N_DRAWS):  # running mean over noise draws
        r, c = run_with_currents(model, inputs, seed)
        r = r[:, :, rec_units] / N_DRAWS
        c = {k: v[:, :, rec_units] / N_DRAWS for k, v in c.items()}
        if rates is None:
            rates, currents = r, c
        else:
            rates += r
            for k in currents:
                currents[k] += c[k]

    # 1. Held-out Poisson log-likelihood
    counts = np.round(np.transpose(data['firing_rates'], (2, 1, 0)) * 0.025)
    test = np.array(config['test_idx'], dtype=np.int64)
    mask = RNNDataset(data).get_all_trials()['mask'].numpy()
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    ll = poisson_ll(rates[test], counts[test], mask[test], 0.025)

    # 2. Effects on model rates, same statistics as for the recordings
    onset = get_target_onset_bins(data)
    model_sess = dict(sess)
    model_sess['vis'] = compute_window_rates(rates, onset)
    model_sess['pre'] = compute_window_rates(rates, onset, window_ms=A.PRE_WINDOW_MS)
    model_eff = effects(model_sess)

    # Recorded effects from the test trials alone and from the rest. Models fit to window
    # means can match training-trial noise; the test-trial effect is the held-out target.
    rec_split = {}
    if len(test):
        is_test = np.isin(np.arange(len(onset)), test)
        for name, keep in [('test', is_test), ('rest', ~is_test)]:
            part = dict(sess)
            part['vis'] = np.where(keep[:, None], sess['vis'], np.nan)
            part['pre'] = np.where(keep[:, None], sess['pre'], np.nan)
            rec_split[name] = effects(part, min_trials=2)  # test trials are few

    # 3. Current decomposition (recorded E units; pre-target window; pref location rewarded vs not)
    decomposition = {}
    is_e_unit = rec_units < n_exc
    reward_map = sess['context']['reward_map']
    for name, cur in currents.items():
        window = compute_window_rates(cur, onset, window_ms=A.PRE_WINDOW_MS)
        diffs = []
        for j in np.flatnonzero(is_e_unit):
            gain = location_rewarded(reward_map, sess['preferred']['for_factors'][j])
            diffs.append(np.nanmean(window[gain, j]) - np.nanmean(window[~gain, j]))
        decomposition[name] = np.array(diffs)

    return {'config': {k: config[k] for k in ('session', 'context', 'context_targets', 'ie_mode', 'hidden_i', 'seed')},
            'll': ll, 'effects': model_eff, 'decomposition': decomposition, 'recorded_split': rec_split}


def main():
    parser = argparse.ArgumentParser(description='Evaluate context models')
    parser.add_argument('--root', default='results/context_pilot')
    args = parser.parse_args()
    root = Path(args.root)

    sessions = {s: A.load_session(s) for s in A.SESSIONS}
    recorded_eff = {s: effects(sess) for s, sess in sessions.items()}

    results = {}
    for model_dir in sorted(p.parent for p in root.glob('*/*/summary.json')):
        try:
            results[model_dir.relative_to(root).as_posix()] = evaluate(model_dir, sessions)
            print(f"evaluated {model_dir}", flush=True)
        except Exception as e:  # report and continue
            print(f"FAILED {model_dir}: {e}", flush=True)

    # Summaries per variant (pooled over sessions, mean over seeds)
    variants = sorted({k.split('/')[0] for k in results})
    baseline = 'none_free'
    lines = [f'# Context model evaluation ({root})', '',
             'Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability '
             'component (sp/s; negative = weaker response at the frequent location), spatial '
             "(pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and "
             'recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson '
             f'log-likelihood per neuron per bin relative to `{baseline}` (x1000).', '',
             '| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r '
             '| Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |',
             '|---|---|---|---|---|---|---|---|']
    rec_pooled = {k: np.concatenate([recorded_eff[s][k] for s in A.SESSIONS]) for k in recorded_eff['08_13']}

    def pooled(variant, seed_key):
        runs = [results[f'{variant}/{s}_seed{seed_key}'] for s in A.SESSIONS
                if f'{variant}/{s}_seed{seed_key}' in results]
        return runs if len(runs) == len(A.SESSIONS) else None

    seeds_all = sorted({k.split('_seed')[-1] for k in results})
    summary = {}
    for v in variants:
        per_seed = []
        for sd in seeds_all:
            runs, base = pooled(v, sd), pooled(baseline, sd)
            if runs is None:
                continue
            entry = {}
            if base is not None:
                entry['dll'] = 1000 * float(np.mean(np.concatenate([r['ll'] for r in runs])
                                                    - np.concatenate([b['ll'] for b in base])))
            for k in ('reward_pre', 'block_pre', 'prob_vis', 'spatial'):
                m = np.concatenate([r['effects'][k] for r in runs])
                ok = ~np.isnan(m) & ~np.isnan(rec_pooled[k])
                entry[f'{k}_model'] = float(np.nanmean(m))
                entry[f'{k}_r'] = float(np.corrcoef(m[ok], rec_pooled[k][ok])[0, 1]) if ok.sum() > 3 else np.nan
                if all(r.get('recorded_split') for r in runs):
                    rt = np.concatenate([r['recorded_split']['test'][k] for r in runs])
                    rr = np.concatenate([r['recorded_split']['rest'][k] for r in runs])
                    ok = ~np.isnan(m) & ~np.isnan(rt) & ~np.isnan(rr)
                    if ok.sum() > 3:
                        entry[f'{k}_r_test'] = float(np.corrcoef(m[ok], rt[ok])[0, 1])
                        entry[f'{k}_ceiling_test'] = float(np.corrcoef(rr[ok], rt[ok])[0, 1])
            dec = {c: np.concatenate([r['decomposition'][c] for r in runs if c in r['decomposition']])
                   for c in runs[0]['decomposition']}
            total_abs = sum(np.abs(d) for d in dec.values())
            entry['ie_share'] = float(np.nanmean(np.abs(dec['IE']) / (total_abs + 1e-9)))
            entry['decomposition_mean'] = {c: float(np.nanmean(d)) for c, d in dec.items()}
            per_seed.append(entry)
        if not per_seed:
            continue
        avg = {k: float(np.nanmean([e.get(k, np.nan) for e in per_seed])) for k in per_seed[0] if k != 'decomposition_mean'}
        summary[v] = {'n_seeds': len(per_seed), 'mean': avg, 'per_seed': per_seed}
        rec = lambda k: np.nanmean(rec_pooled[k])
        lines.append(
            f"| {v} | {len(per_seed)} | {avg.get('dll', np.nan):+.2f} | "
            f"{avg['reward_pre_model']:+.2f} / {rec('reward_pre'):+.2f} / {avg['reward_pre_r']:+.2f} | "
            f"{avg['block_pre_model']:+.2f} / {rec('block_pre'):+.2f} / {avg['block_pre_r']:+.2f} | "
            f"{avg['prob_vis_model']:+.2f} / {rec('prob_vis'):+.2f} / {avg['prob_vis_r']:+.2f} | "
            f"{avg['spatial_r']:+.2f} | {avg['ie_share']:.2f} |")

    lines += ['', 'Held-out check: r between model effect and the recorded effect on test trials only, '
              'and the ceiling r(recorded non-test trials, recorded test trials).', '',
              '| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling '
              '| Spatial r / ceiling |', '|---|---|---|---|---|']
    for v, e in summary.items():
        avg = e['mean']
        if 'spatial_r_test' in avg:
            lines.append(f"| {v} | " + ' | '.join(
                f"{avg.get(k + '_r_test', np.nan):+.2f} / {avg.get(k + '_ceiling_test', np.nan):+.2f}"
                for k in ('reward_pre', 'block_pre', 'prob_vis', 'spatial')) + ' |')

    report = '\n'.join(lines) + '\n'
    (root / 'evaluation_report.md').write_text(report, encoding='utf-8')
    with open(root / 'evaluation_summary.json', 'w') as f:
        json.dump(summary, f, indent=1)
    np.save(root / 'evaluation_raw.npy', results, allow_pickle=True)
    sys.stdout.reconfigure(errors='replace')
    print('\n' + report)


if __name__ == '__main__':
    main()
