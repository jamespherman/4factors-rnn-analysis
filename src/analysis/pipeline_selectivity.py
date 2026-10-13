"""
Visual-epoch factor selectivity matched to the neuronal analysis pipeline.

Reproduces the window ROC analysis of 4factors-analysis-pipeline
(code/analyze_window_roc.m, code/compute_preferred_locations.m,
code/utils/define_task_conditions.m) on the RNN export format, so recorded
neurons and their RNN units can be compared with identical trial sets:

- Window: 50-250 ms after target onset, located per trial (target onset
  varies by ~0.5 s across trials relative to the fixation-aligned time axis)
- Trials restricted to each neuron's preferred target location
- Factor contrasts (cond1 vs cond2; AUC > 0.5 means higher rates for cond2):
    reward:      low vs high reward, all stimuli
    salience:    low vs high salience, bullseye trials only
    identity:    non-face vs face, image trials only
    probability: low vs high probability, at location 1 or 3
- At least 5 trials per condition

Deviation from the pipeline: the export has no saccade times, so neurons whose
visual responses don't differentiate locations fall back to their visual
maximum (the pipeline first tries the saccade response).

Export trial label coding (scripts/export_for_rnn.m):
    trial_reward:      1 = high (rewardDuration > 200 ms), 0 = low
    trial_salience:    0 = N/A (face/non-face trial), 1 = high, 2 = low
    trial_identity:    1 = face, 2 = non-face, 3 = bullseye
    trial_probability: 1 = high, 0 = low
    trial_location:    1-4
"""

from typing import Dict, Optional, Tuple

import numpy as np

from .roc_selectivity import compute_roc_auc

FACTORS = ('reward', 'salience', 'identity', 'probability')

VISUAL_WINDOW_MS = (50.0, 250.0)
MIN_TRIALS = 5
DIFFERENTIATION_RATIO = 1.2

# Opposite location for each target location (pipeline convention)
OPPOSITE_LOCATION = {1: 3, 3: 1, 2: 4, 4: 2}


def get_target_onset_bins(data: dict) -> np.ndarray:
    """
    Per-trial target onset bin, recovered from the target-location input channel.

    export_for_rnn.m switches input_target_loc on at the first bin whose center is
    >= (targetOn - fixOn), so the first nonzero bin is the onset bin.

    Args:
        data: Loaded export (needs 'input_target_loc': [4, n_bins, n_trials])

    Returns:
        onset_bins: [n_trials] int, -1 where the window can't be placed
            (targetOn is NaN, or the target was already on at the first bin,
            i.e. it appeared before the exported window starts)
    """
    target_loc = np.asarray(data['input_target_loc'])
    assert target_loc.ndim == 3 and target_loc.shape[0] == 4, \
        f"Expected input_target_loc [4, n_bins, n_trials], got {target_loc.shape}"

    target_any = target_loc.sum(axis=0) > 0  # [n_bins, n_trials]
    onset_bins = target_any.argmax(axis=0)
    onset_bins[~target_any.any(axis=0)] = -1
    onset_bins[onset_bins == 0] = -1
    return onset_bins


def compute_window_rates(
    rates: np.ndarray,
    onset_bins: np.ndarray,
    bin_size_ms: float = 25.0,
    window_ms: Tuple[float, float] = VISUAL_WINDOW_MS
) -> np.ndarray:
    """
    Average firing rates within a target-aligned window on each trial.

    The onset bin's center lies 0-25 ms after targetOn, so with 25 ms bins and the
    default window, bins onset+2 .. onset+9 span 50-250 ms post-target on average.

    Args:
        rates: [n_trials, n_bins, n_neurons]
        onset_bins: [n_trials] target onset bin per trial (-1 = invalid)
        bin_size_ms: Bin size in ms
        window_ms: (start, end) of window relative to target onset, in ms

    Returns:
        window_rates: [n_trials, n_neurons], NaN for trials without a valid window
    """
    assert rates.ndim == 3, f"Expected rates [n_trials, n_bins, n_neurons], got {rates.shape}"
    n_trials, n_bins, n_neurons = rates.shape
    assert len(onset_bins) == n_trials, "onset_bins must have one entry per trial"

    start_offset = int(round(window_ms[0] / bin_size_ms))
    end_offset = int(round(window_ms[1] / bin_size_ms))

    window_rates = np.full((n_trials, n_neurons), np.nan)
    for t in range(n_trials):
        if onset_bins[t] < 0:
            continue
        start = onset_bins[t] + start_offset
        end = onset_bins[t] + end_offset
        if start < 0 or end > n_bins:
            continue
        window_rates[t] = np.nanmean(rates[t, start:end], axis=0)

    return window_rates


def compute_preferred_locations(
    window_rates: np.ndarray,
    trial_location: np.ndarray,
    trial_mask: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Preferred target location per neuron from visual-epoch responses.

    Args:
        window_rates: [n_trials, n_neurons] visual-epoch rates
        trial_location: [n_trials] target location (1-4)
        trial_mask: [n_trials] bool, trials to use (default: all)

    Returns:
        Dict with:
            'for_factors': [n_neurons] location (1-4) for reward/salience/identity
            'for_probability': [n_neurons] location (1 or 3) for probability
            'location_responses': [n_neurons, 4] mean visual response per location
    """
    n_trials, n_neurons = window_rates.shape
    if trial_mask is None:
        trial_mask = np.ones(n_trials, dtype=bool)

    location_responses = np.full((n_neurons, 4), np.nan)
    for loc in range(1, 5):
        loc_trials = trial_mask & (trial_location == loc)
        if loc_trials.any():
            location_responses[:, loc - 1] = np.nanmean(window_rates[loc_trials], axis=0)

    for_factors = np.full(n_neurons, 1, dtype=int)
    for_probability = np.full(n_neurons, 1, dtype=int)

    for i in range(n_neurons):
        resp = location_responses[i]
        if np.all(np.isnan(resp)):
            continue
        for_factors[i] = int(np.nanargmax(resp)) + 1

        if for_factors[i] in (1, 3):
            for_probability[i] = for_factors[i]
        elif np.isnan(resp[0]):
            for_probability[i] = 3
        elif np.isnan(resp[2]):
            for_probability[i] = 1
        else:
            for_probability[i] = 1 if resp[0] >= resp[2] else 3

    return {
        'for_factors': for_factors,
        'for_probability': for_probability,
        'location_responses': location_responses,
    }


def get_factor_masks(data: dict) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Condition masks (cond1, cond2) for each factor, before location selection.

    Args:
        data: Loaded export with trial_* label fields

    Returns:
        Dict factor -> (cond1_mask, cond2_mask), each [n_trials] bool
    """
    reward = np.asarray(data['trial_reward']).ravel()
    salience = np.asarray(data['trial_salience']).ravel()
    identity = np.asarray(data['trial_identity']).ravel()
    probability = np.asarray(data['trial_probability']).ravel()

    is_bullseye = identity == 3
    is_image = (identity == 1) | (identity == 2)

    return {
        'reward': (reward == 0, reward == 1),
        'salience': (is_bullseye & (salience == 2), is_bullseye & (salience == 1)),
        'identity': (is_image & (identity == 2), is_image & (identity == 1)),
        'probability': (probability == 0, probability == 1),
    }


def compute_factor_selectivity(
    window_rates: np.ndarray,
    data: dict,
    preferred: Dict[str, np.ndarray],
    trial_mask: Optional[np.ndarray] = None,
    min_trials: int = MIN_TRIALS
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Window ROC selectivity for each factor at each neuron's preferred location.

    Args:
        window_rates: [n_trials, n_neurons] visual-epoch rates
        data: Loaded export with trial_* label fields
        preferred: Output of compute_preferred_locations (one entry per neuron)
        trial_mask: [n_trials] bool, trials to use (e.g. held-out trials)
        min_trials: Minimum trials per condition

    Returns:
        Dict factor -> {
            'selectivity': [n_neurons] 2 * (AUC - 0.5), NaN if too few trials
            'auc': [n_neurons]
            'n_trials': [n_neurons, 2]
        }
    """
    n_trials, n_neurons = window_rates.shape
    if trial_mask is None:
        trial_mask = np.ones(n_trials, dtype=bool)
    trial_location = np.asarray(data['trial_location']).ravel()
    assert len(trial_location) == n_trials, "Labels and rates disagree on trial count"

    results = {}
    for factor, (cond1, cond2) in get_factor_masks(data).items():
        pref_key = 'for_probability' if factor == 'probability' else 'for_factors'
        auc = np.full(n_neurons, np.nan)
        counts = np.zeros((n_neurons, 2), dtype=int)

        for i in range(n_neurons):
            loc_mask = trial_mask & (trial_location == preferred[pref_key][i])
            x1 = window_rates[cond1 & loc_mask, i]
            x2 = window_rates[cond2 & loc_mask, i]
            x1, x2 = x1[~np.isnan(x1)], x2[~np.isnan(x2)]
            counts[i] = (len(x1), len(x2))
            if len(x1) >= min_trials and len(x2) >= min_trials:
                auc[i] = compute_roc_auc(x1, x2)

        results[factor] = {'selectivity': 2 * (auc - 0.5), 'auc': auc, 'n_trials': counts}

    return results


def compute_location_selectivity(
    window_rates: np.ndarray,
    trial_location: np.ndarray,
    preferred_location: np.ndarray,
    trial_mask: Optional[np.ndarray] = None,
    min_trials: int = MIN_TRIALS
) -> np.ndarray:
    """
    Spatial selectivity: preferred vs opposite target location.

    To avoid circularity, preferred_location should be estimated on trials
    disjoint from trial_mask.

    Args:
        window_rates: [n_trials, n_neurons] visual-epoch rates
        trial_location: [n_trials] target location (1-4)
        preferred_location: [n_neurons] preferred location (1-4)
        trial_mask: [n_trials] bool, trials to use
        min_trials: Minimum trials per condition

    Returns:
        [n_neurons] 2 * (AUC - 0.5), positive = higher at preferred location
    """
    n_trials, n_neurons = window_rates.shape
    if trial_mask is None:
        trial_mask = np.ones(n_trials, dtype=bool)

    selectivity = np.full(n_neurons, np.nan)
    for i in range(n_neurons):
        pref = int(preferred_location[i])
        x_pref = window_rates[trial_mask & (trial_location == pref), i]
        x_opp = window_rates[trial_mask & (trial_location == OPPOSITE_LOCATION[pref]), i]
        x_pref, x_opp = x_pref[~np.isnan(x_pref)], x_opp[~np.isnan(x_opp)]
        if len(x_pref) >= min_trials and len(x_opp) >= min_trials:
            selectivity[i] = 2 * (compute_roc_auc(x_opp, x_pref) - 0.5)

    return selectivity
