#!/usr/bin/env python3
"""
Visual-epoch ROC selectivity of recorded neurons and their RNN units, by monkey and cell type.

For each neuron and factor: selectivity = 2 x (ROC area - 0.5), 50-250 ms after target onset,
at the neuron's preferred location, on all verified trials (each neuron only within its stable
period), as in compare_recorded_vs_rnn_selectivity.py. RNN rates go through Poisson spike sampling
and each noise draw is scored on its own, so RNN and recorded selectivity carry comparable
sampling noise. Significance is a label permutation test (two-sided p < 0.05).

Location compares preferred vs opposite location, preferred estimated on even trials and scored
on odd trials.

Usage:
    python scripts/compare_selectivity_by_type.py --grid results/grid_v13 --variant free_free
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src.sessions import ALL_SESSIONS, load_export, stable_mask
from src.analysis.pipeline_selectivity import (
    get_target_onset_bins, compute_window_rates, compute_preferred_locations, get_factor_masks,
    OPPOSITE_LOCATION, MIN_TRIALS)
from evaluate_context_models import load_context_model, build_inputs, run_with_currents
from compare_recorded_vs_rnn_selectivity import sample_spike_rates

BASE_DIR = Path(__file__).parent.parent
FACTORS = ['location', 'identity', 'salience', 'reward', 'probability']


def contrasts(data: dict, recorded_rates: np.ndarray) -> dict:
    """Per factor, per neuron (cond1 trial mask, cond2 trial mask); selectivity > 0 means cond2 higher."""
    n_trials, n_neurons = recorded_rates.shape
    loc = np.asarray(data['trial_location']).ravel()
    pref = compute_preferred_locations(recorded_rates, loc)
    even = np.arange(n_trials) % 2 == 0
    loc_pref = compute_preferred_locations(recorded_rates, loc, even)['for_factors']
    out = {'location': [(~even & (loc == OPPOSITE_LOCATION[int(p)]), ~even & (loc == p)) for p in loc_pref]}
    for factor, (c1, c2) in get_factor_masks(data).items():
        key = 'for_probability' if factor == 'probability' else 'for_factors'
        out[factor] = [(c1 & (loc == p), c2 & (loc == p)) for p in pref[key]]
    return out


def perms_for(n: int, n_perm: int) -> np.ndarray:
    """[n_perm, n] fixed random permutations of n trials (cached per n)."""
    if (n, n_perm) not in _PERM_CACHE:
        rng = np.random.default_rng(n)
        _PERM_CACHE[(n, n_perm)] = np.stack([rng.permutation(n) for _ in range(n_perm)])
    return _PERM_CACHE[(n, n_perm)]


_PERM_CACHE = {}


def roc_with_null(x: np.ndarray, m1: np.ndarray, m2: np.ndarray, n_perm: int):
    """Selectivity (cond2 > cond1 positive) and two-sided label-permutation p for one neuron."""
    use = (m1 | m2) & ~np.isnan(x)
    if (m1 & use).sum() < MIN_TRIALS or (m2 & use).sum() < MIN_TRIALS:
        return np.nan, np.nan
    idx = np.flatnonzero(use)
    r = rankdata(x[idx])
    lab = m2[idx].astype(float)
    n2, n1 = lab.sum(), len(lab) - lab.sum()

    def sel(rank_sum):
        return 2 * ((rank_sum - n2 * (n2 + 1) / 2) / (n1 * n2) - 0.5)

    obs = sel(lab @ r)
    null = sel(lab[perms_for(len(idx), n_perm)] @ r)
    return obs, (np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (n_perm + 1)


def score(rates: np.ndarray, con: dict, n_perm: int) -> dict:
    """[n_neurons] selectivity and p per factor."""
    out = {}
    for f in FACTORS:
        res = [roc_with_null(rates[:, i], m1, m2, n_perm) for i, (m1, m2) in enumerate(con[f])]
        out[f] = np.array(res).reshape(-1, 2)
    return out


def main():
    parser = argparse.ArgumentParser(description='Recorded vs RNN ROC selectivity by monkey and cell type')
    parser.add_argument('--grid', default='results/grid_v13')
    parser.add_argument('--variant', default='free_free')
    parser.add_argument('--restarts', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6])
    parser.add_argument('--n-draws', type=int, default=5)
    parser.add_argument('--n-perm', type=int, default=500)
    args = parser.parse_args()

    rows = {'session': [], 'neuron_type': []}
    rec = {f: [] for f in FACTORS}
    rnn = {f: [] for f in FACTORS}  # each [n_neurons, restarts, draws, 2]
    for s in ALL_SESSIONS:
        print(f'{s}...', flush=True)
        cfg0 = json.load(open(BASE_DIR / args.grid / args.variant / f'{s}_seed{args.restarts[0]}' / 'config.json'))
        data = load_export(cfg0['session'], cfg0['trials'])
        stable = stable_mask(data).T
        onset = get_target_onset_bins(data)
        recorded = compute_window_rates(np.transpose(data['firing_rates'], (2, 1, 0)), onset)
        con = contrasts(data, recorded)
        nt = np.asarray(data['neuron_type']).ravel()
        rows['session'] += [s] * len(nt)
        rows['neuron_type'] += ['E' if t == 1 else 'I' for t in nt]
        sc = score(recorded, con, args.n_perm)
        for f in FACTORS:
            rec[f].append(sc[f])
        per_restart = {f: [] for f in FACTORS}
        for k in args.restarts:
            model, config = load_context_model(BASE_DIR / args.grid / args.variant / f'{s}_seed{k}')
            units = np.array(config['recorded_unit_indices'])
            assert np.array_equal(units < config['n_exc'], nt == 1), 'unit type differs from neuron type'
            inputs = build_inputs(data, config)
            draws = {f: [] for f in FACTORS}
            for d in range(args.n_draws):
                rates = sample_spike_rates(run_with_currents(model, inputs, d)[0][:, :, units], seed=d)
                sc = score(np.where(stable, compute_window_rates(rates, onset), np.nan), con, args.n_perm)
                for f in FACTORS:
                    draws[f].append(sc[f])
            for f in FACTORS:
                per_restart[f].append(np.stack(draws[f], 1))
        for f in FACTORS:
            rnn[f].append(np.stack(per_restart[f], 1))

    out = BASE_DIR / 'results' / 'selectivity_by_type'
    out.mkdir(parents=True, exist_ok=True)
    stem = f'{Path(args.grid).name}_{args.variant}'
    np.savez(out / f'{stem}.npz', session=np.array(rows['session']), neuron_type=np.array(rows['neuron_type']),
             **{f'recorded_{f}': np.concatenate(rec[f]) for f in FACTORS},
             **{f'rnn_{f}': np.concatenate(rnn[f]) for f in FACTORS})
    print(f'saved {out / stem}.npz')


if __name__ == '__main__':
    main()
