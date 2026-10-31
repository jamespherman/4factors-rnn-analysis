"""
Shared per-session tables for the inter-collicular analyses (plans/2026-10-31_intercollicular_plan.md).

For each session: spike times (src.spikes), the export's trial labels, each export location's polar angle and
hemifield relative to the recorded SC (from the pipeline's cached screening groups), and per-trial window rates.
"""

import itertools
import sys
from pathlib import Path
from typing import Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
from sc_spatial_profile import load_cache  # noqa: E402

from src.sessions import load_export, session_name  # noqa: E402
from src.spikes import load_spikes  # noqa: E402

# Windows in s: (event, start, end)
WINDOWS = {
    'baseline': ('ev_targetOn', -0.2, 0.0),
    'visual': ('ev_targetOn', 0.05, 0.25),
    'presac': ('ev_saccadeOnset', -0.05, 0.0),
    'postsac': ('ev_saccadeOnset', 0.0, 0.1),
}
OPPOSITE = {1: 3, 3: 1, 2: 4, 4: 2}


def counts(spk: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Spike counts in [start, end) per trial (NaN where start is NaN)."""
    out = (np.searchsorted(spk, ends) - np.searchsorted(spk, starts)).astype(float)
    out[np.isnan(starts) | np.isnan(ends)] = np.nan
    return out


def session_table(session: str) -> Dict:
    """Everything the Phase 1 analyses need for one session."""
    sp = load_spikes(session)
    data = load_export(session, 'verified', stable=False)
    cache = load_cache(session_name(session))
    ids = sp['ids']
    loc = np.asarray(data['trial_location']).ravel().astype(int)
    n_trials, n_units = len(loc), len(ids)
    stable = sp['stable'].T                                                     # [trials, units]

    rates = {}
    for w, (ev, a, b) in WINDOWS.items():
        r = np.stack([counts(sp[f'spk_{i}'], sp[ev] + a, sp[ev] + b) / (b - a) for i in ids], 1)
        rates[w] = np.where(stable, r, np.nan)

    # Export location -> cache group by visual-epoch means (assignment varies by session)
    col = np.array([np.flatnonzero(cache['cluster_ids'] == i)[0] for i in ids])
    ev = np.stack([np.nanmean(rates['visual'][loc == k + 1], 0) for k in range(4)])
    ev -= ev.mean(0)
    best = max(((np.corrcoef(ev.ravel(), (cache['means'][list(q)][:, col, 1]
                                          - cache['means'][list(q)][:, col, 1].mean(0)).ravel())[0, 1], q)
                for q in itertools.permutations(range(4))))
    assert best[0] > 0.9, f'{session}: location mapping r={best[0]:.2f}'
    angle = cache['angles'][list(best[1])]                                       # angle of export loc 1..4
    sign = -1 if cache['side'] == 'right' else 1
    contra_x = np.cos(np.deg2rad(angle)) * sign                                  # > 0 contralateral
    hemi = np.where(np.abs(contra_x) < 0.05, 'vertical', np.where(contra_x > 0, 'contra', 'ipsi'))

    return {
        'session': session, 'monkey': 'Feynman' if session.startswith('Feynman') else 'Newton',
        'ids': ids, 'type': np.where(sp['neuron_type'] == 1, 'E', 'I'), 'depth': sp['info_depth'],
        'ch': sp['info_ch'], 'wf': sp['wf'], 'stable': stable, 'rates': rates, 'loc': loc,
        'angle': angle, 'hemi': hemi, 'elev': np.sign(np.round(np.sin(np.deg2rad(angle)), 6)),
        'reward': np.asarray(data['trial_reward']).ravel(), 'probability': np.asarray(data['trial_probability']).ravel(),
        'identity': np.asarray(data['trial_identity']).ravel(), 'salience': np.asarray(data['trial_salience']).ravel(),
        'spikes': {i: sp[f'spk_{i}'] for i in ids},
        'ev': {k[3:]: sp[k] for k in sp if k.startswith('ev_')},
        'rt': sp['ev_saccadeOnset'] - sp['ev_fixOff'],
    }
