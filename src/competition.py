"""
Trial table for the competition analyses (plans/2026-10-31_competition_plan.md): every gSac_4factors trial
(taskCode 32015) of a session from the pipeline's session_data.mat, including error trials, with block (which
location is high-probability), target, factors, reaction time, peak velocity and saccade endpoint, and the index
of the trial in the v13 export (-1 if not exported).
"""

from pathlib import Path
from typing import Dict

import h5py
import numpy as np

from src.sessions import session_name, load_export
from src.spikes import PIPELINE

CACHE = Path(__file__).parent.parent / 'data' / 'spike_cache'
TASK = 32015
HIGH_P_LOC = {1: 1, 2: 3}    # block -> high-probability target location index (export_for_rnn.m)


def trial_table(session: str) -> Dict[str, np.ndarray]:
    name = session_name(session)
    path = CACHE / f'{name}_trials.npz'
    if path.exists():
        d = np.load(path)
        return {k: d[k] for k in d.files}
    x = load_export(session, 'verified', stable=False)
    exp_rows = np.asarray(x['trial_session_index']).ravel().astype(int) - 1
    with h5py.File(PIPELINE / name / f'{name}_session_data.mat', 'r') as f:
        ti, ev = f['session_data/trialInfo'], f['session_data/eventTimes']
        g = lambda k: np.asarray(ti[k]).ravel()
        task = g('taskCode')
        rows = np.flatnonzero(task == TASK)
        out = {
            'row': rows, 'block': g('blockNumber')[rows], 'loc': g('targetLocIdx')[rows],
            'theta': g('targetTheta')[rows] / 10, 'end_state': g('trialEndState')[rows],
            'stim_type': g('stimType')[rows], 'salience': g('salience')[rows],
            'reward_dur': g('rewardDuration')[rows], 'peak_vel': g('peakVel')[rows],
            'post_x': np.asarray(ti['postSacXY'])[0, rows] if ti['postSacXY'].shape[0] == 2 else np.asarray(ti['postSacXY'])[rows, 0],
            'post_y': np.asarray(ti['postSacXY'])[1, rows] if ti['postSacXY'].shape[0] == 2 else np.asarray(ti['postSacXY'])[rows, 1],
            **{e: np.asarray(ev[e]).ravel()[rows] for e in ('fixOn', 'targetOn', 'fixOff', 'saccadeOnset')},
        }
    out['rt'] = out['saccadeOnset'] - out['fixOff']
    out['high_p_loc'] = np.array([HIGH_P_LOC.get(int(b), -1) if not np.isnan(b) else -1 for b in out['block']])
    pos = {r: k for k, r in enumerate(exp_rows)}
    out['export_idx'] = np.array([pos.get(r, -1) for r in rows])
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    return out
