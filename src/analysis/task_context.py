"""
Block context of the 4factors task, recovered from the RNN export labels.

Task design (verified against raw trialInfo of the neuronal pipeline):
    Probability: block 1 -> location 1 is the frequent target, block 2 -> location 3.
        Locations 2 and 4 are always low probability.
    Reward: deterministic given half-block and location. Half-blocks 1 and 3:
        locations 1 and 2 give a large reward; half-blocks 2 and 4: locations 3 and 4.

The export has no block fields, but every trial's (location, reward) gives the
reward map, and (location, probability) on location 1 and 3 trials gives the block.
Block on location 2 and 4 trials is filled from the nearest known trial within the
same reward-map run.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

PIPELINE_PROCESSED = (Path(__file__).parent.parent.parent.parent
                      / '4factors-analysis-pipeline' / 'data' / 'processed')


def get_task_context(data: dict) -> Dict[str, np.ndarray]:
    """
    Per-trial block context from export labels.

    Args:
        data: Loaded export with trial_location, trial_reward, trial_probability

    Returns:
        Dict with:
            'reward_map': [n_trials] int, 1 = locations 1, 2 rewarded (half-blocks 1, 3),
                0 = locations 3, 4 rewarded (half-blocks 2, 4)
            'block': [n_trials] float, 1 or 2 (NaN if it can't be filled)
            'reward_run': [n_trials] int, index of the reward-map run
    """
    location = np.asarray(data['trial_location']).ravel().astype(int)
    reward = np.asarray(data['trial_reward']).ravel().astype(int)
    probability = np.asarray(data['trial_probability']).ravel().astype(int)

    reward_map = (np.isin(location, [1, 2]) == (reward == 1)).astype(int)
    reward_run = np.r_[0, np.cumsum(np.diff(reward_map) != 0)]

    block = np.full(len(location), np.nan)
    block[(location == 1) & (probability == 1)] = 1
    block[(location == 3) & (probability == 0)] = 1
    block[(location == 3) & (probability == 1)] = 2
    block[(location == 1) & (probability == 0)] = 2

    known = ~np.isnan(block)
    filled = block.copy()
    for t in np.flatnonzero(~known):
        candidates = np.flatnonzero(known & (reward_run == reward_run[t]))
        if len(candidates):
            filled[t] = block[candidates[np.argmin(np.abs(candidates - t))]]

    return {'reward_map': reward_map, 'block': filled, 'reward_run': reward_run}


def location_rewarded(reward_map: np.ndarray, location: int) -> np.ndarray:
    """[n_trials] bool: is this location currently the large-reward location?"""
    return (reward_map == 1) if location in (1, 2) else (reward_map == 0)


def location_frequent(block: np.ndarray, location: int) -> Optional[np.ndarray]:
    """[n_trials] bool: is this location currently the frequent one? None for 2 and 4."""
    if location == 1:
        return block == 1
    if location == 3:
        return block == 2
    return None


def get_switches(state: np.ndarray) -> np.ndarray:
    """Trial indices where a context state changes (first trial of the new state)."""
    state = np.asarray(state, dtype=float)
    switches = np.flatnonzero(np.diff(state) != 0) + 1
    return np.array([t for t in switches if not np.isnan(state[t]) and not np.isnan(state[t - 1])],
                    dtype=int)


def match_raw_trials(data: dict, session_name: str) -> Optional[Dict[str, np.ndarray]]:
    """
    Match export trials to the pipeline's raw trialInfo (rewarded trials, in order),
    on location, reward, identity and salience.

    Returns:
        Dict with raw 'halfBlock', 'blockNumber', 'targetLocIdx' for all raw trials and
        'export_to_raw' [n_export] raw trial index, or None if the session file is
        missing or matching fails.
    """
    import h5py

    path = PIPELINE_PROCESSED / session_name / f'{session_name}_session_data.mat'
    if not path.exists():
        return None
    with h5py.File(path, 'r') as f:
        info = f['session_data/trialInfo']
        get = lambda k: np.array(info[k]).ravel()
        reward_duration, loc, stim_type, salience = (get('rewardDuration'), get('targetLocIdx'),
                                                     get('stimType'), get('salience'))
        half_block, block_number = get('halfBlock'), get('blockNumber')

    identity = np.where(stim_type == 1, 1, np.where(stim_type == 2, 2, 3))
    rewarded = np.flatnonzero((reward_duration > 0) & ~np.isnan(loc))
    raw_keys = np.stack([loc, reward_duration > 200, identity, salience], axis=1)[rewarded]
    export_keys = np.stack([np.asarray(data[k]).ravel() for k in
                            ('trial_location', 'trial_reward', 'trial_identity', 'trial_salience')], axis=1)

    raw_keys = np.nan_to_num(raw_keys, nan=-1).astype(int)
    export_keys = export_keys.astype(int)
    j, export_to_raw = 0, []
    for t in range(len(export_keys)):
        while j < len(raw_keys) and not np.array_equal(raw_keys[j], export_keys[t]):
            j += 1
        if j == len(raw_keys):
            return None
        export_to_raw.append(rewarded[j])
        j += 1

    return {'halfBlock': half_block, 'blockNumber': block_number, 'targetLocIdx': loc,
            'export_to_raw': np.array(export_to_raw)}


TRIAL_AUDIT_DIR = Path(__file__).parent.parent.parent / 'data' / 'trial_audit'


def get_verified_success(session_name: str, n_trials: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Successful export trials with a verified PLDAPS match, and their true reward times.

    The export keeps every trial with a 'reward' strobe, but that code is also sent at the
    end of every trial, so it includes fixation breaks and trials whose event times are
    unreliable. Its reward time (= trial_duration_ms and the input_reward_on onset) is the
    end-of-trial strobe dump, about 648 ms (high reward) or 456 ms (low) after the real
    delivery. data/trial_audit/{session}_trial_audit.csv comes from the
    4factors-analysis-pipeline audit (2026-10-13); export_row (1-based) gives the export order.

    Returns:
        keep: [n_trials] bool, trialEndState 21 and match_verified
        true_reward_ms: [n_trials] real reward delivery relative to fixOn (the export's
            t = 0); only meaningful where keep is True
    """
    path = TRIAL_AUDIT_DIR / f'{session_name}_trial_audit.csv'
    assert path.exists(), f"Missing trial audit {path}"
    audit = np.genfromtxt(path, delimiter=',', names=True)
    rows = audit[~np.isnan(audit['export_row'])]
    rows = rows[np.argsort(rows['export_row'])]
    assert len(rows) == n_trials and np.all(rows['export_row'] == np.arange(1, n_trials + 1)),         f"Trial audit doesn't cover the {n_trials} export trials"
    keep = (rows['match_verified'] == 1) & (rows['trialEndState_pipeline'] == 21) & (rows['nev_success'] == 1)
    return keep, rows['true_reward_ms_rel_fixOn']


def apply_true_reward_times(data: dict, true_reward_ms: np.ndarray) -> dict:
    """
    Move the reward input to the real delivery time and end each trial there
    (the export ends trials at its reward time).
    """
    assert not np.any(np.isnan(true_reward_ms)), "True reward time missing for some trials"
    time_axis = np.asarray(data['time_axis']).ravel()
    reward_on = np.asarray(data['input_reward_on'])
    n_bins, n_trials = reward_on.shape
    duration_bins = int((reward_on > 0).sum(axis=0).max())
    shifted = np.zeros_like(reward_on)
    for t in range(n_trials):
        onset = int(np.searchsorted(time_axis, true_reward_ms[t]))
        shifted[onset:onset + duration_bins, t] = 1
    out = dict(data)
    out['input_reward_on'] = shifted
    out['trial_duration_ms'] = np.asarray(true_reward_ms, dtype=float)
    return out
