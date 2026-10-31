"""
Full-resolution spike times, trial events and unit metadata for the exported neurons, read from the
pipeline's processed session_data.mat (MATLAB v7.3) and cached as .npz.

Trials follow the v13 export (verified trials): trial k of the export is row trial_session_index[k] - 1
of session_data.eventTimes. Times are seconds on the spike clock.
"""

from pathlib import Path
from typing import Dict

import h5py
import numpy as np

from src.sessions import load_export, session_name

PIPELINE = Path(r'D:/OneDrive - University of Pittsburgh/Code/4factors-analysis-pipeline/data/processed')
CACHE = Path(__file__).parent.parent / 'data' / 'spike_cache'
EVENTS = ['fixOn', 'targetOn', 'fixOff', 'saccadeOnset', 'saccadeOffset']


def _build(session: str) -> Dict[str, np.ndarray]:
    name = session_name(session)
    data = load_export(session, 'verified', stable=False)
    ids = np.asarray(data['neuron_ids']).ravel().astype(int)
    rows = np.asarray(data['trial_session_index']).ravel().astype(int) - 1
    with h5py.File(PIPELINE / name / f'{name}_session_data.mat', 'r') as f:
        sp = f['session_data/spikes']
        times = np.asarray(sp['times']).ravel()
        clusters = np.asarray(sp['clusters']).ravel().astype(int)
        ci_ids = np.asarray(sp['cluster_info/cluster_id']).ravel().astype(int)
        ci_rows = np.array([np.flatnonzero(ci_ids == i)[0] for i in ids])
        info = {k: np.asarray(sp[f'cluster_info/{k}']).ravel()[ci_rows] for k in ('ch', 'depth', 'amp', 'ContamPct')}
        wf_refs = np.asarray(sp['wfMeans']).ravel()
        assert len(wf_refs) == len(ci_ids), 'wfMeans not aligned with cluster_info'
        wf = np.stack([np.asarray(f[wf_refs[r]]) for r in ci_rows])           # [units, samples, channels]
        ev = f['session_data/eventTimes']
        events = {e: np.asarray(ev[e]).ravel()[rows] for e in EVENTS}
    out = {'ids': ids, 'trial_rows': rows, 'neuron_type': np.asarray(data['neuron_type']).ravel(),
           'stable': np.asarray(data['neuron_stable_trials']).astype(bool),       # [units, trials]
           'trial_location': np.asarray(data['trial_location']).ravel(),
           'wf': wf, **{f'info_{k}': v for k, v in info.items()}, **{f'ev_{e}': v for e, v in events.items()}}
    order = np.argsort(clusters, kind='stable')
    cl, t = clusters[order], times[order]
    for i in ids:
        lo, hi = np.searchsorted(cl, [i, i + 1])
        out[f'spk_{i}'] = np.sort(t[lo:hi])
    return out


def load_spikes(session: str) -> Dict[str, np.ndarray]:
    """Spike times per exported unit ('spk_<cluster id>'), events per export trial ('ev_<name>'), unit info."""
    path = CACHE / f'{session_name(session)}.npz'
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(path, **_build(session))
    d = np.load(path)
    return {k: d[k] for k in d.files}
