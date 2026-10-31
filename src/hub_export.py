"""
Matched sampling and export of simulated srsClassic sessions for bilateral-SC-recordings' pipeline
(plans/2026-10-31_hub_resolver_models_plan.md).

Input (from conflict-task-RNN code/srs simulate()):
    trials: DataFrame, one row per simulated trial, aligned to a recorded trial table (same order), with model outcome
            columns: chosen_side (1 right, 2 left, 0 none), outcome, rt_s (go -> saccade onset), rewarded,
            reward_duration_ms; and per-trial times relative to the start of the rate window (s): rel_target_on,
            rel_go, rel_saccade (NaN if none).
    rates:  [trials, time, units] firing rates (any units), bins of dt seconds starting at the rate window start.
    units:  DataFrame per model unit: side ('left'/'right'), kind ('E_pool', 'E_other', 'I_local', 'I_comm').
Output: trials.csv with the recorded column names on one session clock (recorded event times re-anchored to the
model's saccade), and units.mat (struct array: side, unit_id, label, times).
"""

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.io import savemat

# Recorded unit counts per side and classic (pool-E-like) fraction (bilateral-SC-recordings, classic_labels.csv)
RECORDED = {
    '08-20': {'left': (19, 6), 'right': (27, 0)},
    '08-19': {'left': (39, 0), 'right': (38, 6)},
}
RATE_MEDIAN, RATE_SIGMA = 4.0, 0.8      # lognormal target mean rates: median ~4 sp/s, a few at 15-20 sp/s


def sample_units(units: pd.DataFrame, session: str, rng: np.random.Generator) -> pd.DataFrame:
    """Draw model units per side in the recorded counts, with the recorded number of pool-E (classic-like) units."""
    picks = []
    for side, (n, n_classic) in RECORDED[session].items():
        pool = units.index[(units['side'] == side) & (units['kind'] == 'E_pool')].to_numpy()
        other = units.index[(units['side'] == side) & (units['kind'] != 'E_pool')].to_numpy()
        a = rng.choice(pool, size=n_classic, replace=len(pool) < n_classic)
        b = rng.choice(other, size=n - n_classic, replace=len(other) < n - n_classic)
        picks.append(np.r_[a, b])
    sel = units.loc[np.concatenate(picks)].copy()
    sel['model_index'] = sel.index
    return sel.reset_index(drop=True)


def export_session(trials: pd.DataFrame, rates: np.ndarray, units: pd.DataFrame, recorded: pd.DataFrame,
                   session: str, out_dir: Path, dt: float = 0.01, seed: int = 0) -> Dict[str, Path]:
    """Poisson-sample a matched set of units and write trials.csv + units.mat for one simulated session."""
    rng = np.random.default_rng(seed)
    assert len(trials) == len(recorded) == rates.shape[0], 'trials must align with the recorded trial table'
    sel = sample_units(units, session, rng)
    # Session clock: recorded target-onset times; the rate window starts rel_target_on before target onset
    t0 = recorded['t_target_on'].to_numpy() - trials['rel_target_on'].to_numpy()
    # Per-unit gain so each sampled unit's mean rate matches a recorded-like lognormal target
    m = rates[:, :, sel['model_index']].mean(axis=(0, 1))
    target = np.exp(np.log(RATE_MEDIAN) + RATE_SIGMA * rng.standard_normal(len(sel)))
    gain = target / np.maximum(m, 1e-6)
    times = []
    for u, mi in enumerate(sel['model_index']):
        lam = np.clip(rates[:, :, mi], 0, None) * gain[u] * dt                     # expected counts per bin
        n = rng.poisson(lam)
        tr, b = np.nonzero(n)
        k = n[tr, b]
        tt = np.repeat(t0[tr] + b * dt, k) + rng.uniform(0, dt, k.sum())
        times.append(np.sort(tt))
    # Trial table: recorded columns, with model behaviour and model-timed events
    out = recorded.copy()
    for c in ('chosen_side', 'outcome', 'rewarded', 'reward_duration_ms', 'rt_s'):
        if c in trials:
            out[c] = trials[c].to_numpy()
    out['t_target_on'] = recorded['t_target_on'].to_numpy()
    out['t_go_signal'] = t0 + trials['rel_go'].to_numpy()
    out['t_saccade_onset'] = t0 + trials['rel_saccade'].to_numpy()
    out['t_reward'] = np.where(out['rewarded'] == 1, out['t_saccade_onset'] + 0.3, np.nan)
    out['has_behavior'] = 1
    out_dir.mkdir(parents=True, exist_ok=True)
    p_trials = out_dir / 'trials.csv'
    out.to_csv(p_trials, index=False)
    st = np.zeros((len(sel),), dtype=[('side', 'O'), ('unit_id', 'O'), ('label', 'O'), ('times', 'O')])
    for u in range(len(sel)):
        st[u] = (sel.loc[u, 'side'], f'model_{sel.loc[u, "model_index"]}', sel.loc[u, 'kind'], times[u])
    p_units = out_dir / 'units.mat'
    savemat(p_units, {'units': st}, do_compression=True)
    sel.assign(gain=gain, target_rate=target).to_csv(out_dir / 'units_meta.csv', index=False)
    return {'trials': p_trials, 'units': p_units}
