"""
Session names, export locations and trial selection.

Sessions are named like the pipeline ('Newton_08_13_2025_SC'). Short forms are
accepted everywhere a session is given: '08_13' (a Newton session, the original
three-session set) and 'Feynman_08_12' (monkey + date).

Trial sets:
    'all'       every exported trial
    'verified'  successful trials only, with true event times. Exports that carry a
                per-trial outcome field (planned for the corrected re-export) are
                filtered on it; older exports use data/trial_audit (see
                task_context.get_verified_success), which also moves the reward
                input to the true delivery time.
"""

import os
import re
from pathlib import Path
from typing import List

import numpy as np

from src.data_loader import load_mat_file, select_trials

REPO_DIR = Path(__file__).parent.parent
DATA_DIR = REPO_DIR / 'data'
# Pre-correction exports (pipeline < 1.1.0: fixation breaks included, Newton only). Results
# before 2026-10-14 evening were computed from these; set FOURFACTORS_EXPORT_DIR to rebuild them.
V1_EXPORT_DIR = DATA_DIR / 'export_2026-10-14_v1.0'

# Original three-session set (short ids '08_13' etc. are Newton sessions)
DEFAULT_SESSIONS = ['08_13', '08_14', '08_15']
# Every SC session with a corrected export (pipeline 1.1.0)
ALL_SESSIONS = ['Feynman_08_12', 'Feynman_08_13', 'Feynman_08_14', 'Feynman_08_15', '08_13', '08_14', '08_15']

# Per-neuron stable-trial mask [n_neurons, n_trials] (requested from the pipeline 2026-10-14)
STABLE_FIELD = 'neuron_stable_trials'

# Per-trial outcome fields a corrected export may carry (21 = success)
OUTCOME_FIELDS = ('trial_end_state', 'trialEndState', 'trial_outcome')
SUCCESS_END_STATE = 21


def session_name(session: str) -> str:
    """'08_13' -> 'Newton_08_13_2025_SC'; 'Feynman_08_12' -> 'Feynman_08_12_2025_SC'."""
    if re.fullmatch(r'\d\d_\d\d', session):
        return f'Newton_{session}_2025_SC'
    if re.fullmatch(r'[A-Za-z]+_\d\d_\d\d', session):
        return f'{session}_2025_SC'
    assert re.fullmatch(r'[A-Za-z]+_\d\d_\d\d_\d{4}_\w+', session), f"Unrecognized session: {session}"
    return session


def export_dir() -> Path:
    """Folder of the exports in use: FOURFACTORS_EXPORT_DIR if set, else data/ (corrected exports)."""
    return Path(os.environ.get('FOURFACTORS_EXPORT_DIR', DATA_DIR))


def trial_audit_dir() -> Path:
    """Trial audit CSVs belonging to the exports in use."""
    return export_dir() / 'trial_audit'


def export_path(session: str) -> Path:
    """Export file of a session in the export folder."""
    return export_dir() / f'rnn_export_{session_name(session)}.mat'


def available_sessions() -> List[str]:
    """Full names of the sessions with an export in the export folder."""
    return sorted(p.stem.replace('rnn_export_', '') for p in export_dir().glob('rnn_export_*.mat'))


def outcome_field(data: dict):
    """Name of the export's per-trial outcome field, or None."""
    return next((f for f in OUTCOME_FIELDS if f in data), None)


def stable_mask(data: dict) -> np.ndarray:
    """
    [n_neurons, n_trials] bool, True where the trial lies in the neuron's stable recording
    period. All True when the export has no stable-trial field.
    """
    n_neurons, _, n_trials = np.shape(data['firing_rates'])
    if STABLE_FIELD not in data:
        return np.ones((n_neurons, n_trials), dtype=bool)
    mask = np.asarray(data[STABLE_FIELD]).astype(bool)
    assert mask.shape == (n_neurons, n_trials), f"{STABLE_FIELD} shape {mask.shape}"
    return mask


def mask_unstable(data: dict) -> dict:
    """Set firing rates outside each neuron's stable period to NaN (ignored by losses and analyses)."""
    mask = stable_mask(data)
    if mask.all():
        return data
    out = dict(data)
    out['firing_rates'] = np.where(mask[:, None, :], np.asarray(data['firing_rates'], dtype=float), np.nan)
    return out


def select_trial_set(data: dict, session: str, trials: str) -> dict:
    """Restrict a loaded export to a trial set ('all' or 'verified')."""
    if trials == 'all':
        return data
    assert trials == 'verified', f"Unknown trial set: {trials}"
    field = outcome_field(data)
    if field is not None:
        return select_trials(data, np.asarray(data[field]).ravel() == SUCCESS_END_STATE)
    from src.analysis.task_context import get_verified_success, apply_true_reward_times
    keep, true_reward_ms = get_verified_success(session_name(session), data['firing_rates'].shape[-1])
    return apply_true_reward_times(select_trials(data, keep), true_reward_ms[keep])


def load_export(session: str, trials: str = 'all', path=None, stable: bool = True) -> dict:
    """
    Load a session's export restricted to a trial set.

    Args:
        session: Session name (short forms accepted)
        trials: 'all' or 'verified'
        path: Export file to load instead of the export folder's rnn_export_<name>.mat
        stable: Set firing rates outside each neuron's stable period to NaN
    """
    path = Path(path) if path is not None else export_path(session)
    assert path.exists(), f"Missing export {path}"
    data = select_trial_set(load_mat_file(str(path)), session, trials)
    return mask_unstable(data) if stable else data


PROCESSED_CANDIDATES = [
    # Planned home of the pipeline's data (2026-10-14), then its old location in OneDrive\Code
    Path(r'D:\OneDrive - University of Pittsburgh\Neuronal Data Analysis\4factors-analysis-pipeline\data\processed'),
    Path(r'D:\OneDrive - University of Pittsburgh\Code\4factors-analysis-pipeline\data\processed'),
]


def processed_dir() -> Path:
    """
    The 4factors-analysis-pipeline's data/processed folder (raw trialInfo): the
    FOURFACTORS_PROCESSED_DIR environment variable if set, else the first existing
    PROCESSED_CANDIDATES entry.
    """
    if 'FOURFACTORS_PROCESSED_DIR' in os.environ:
        return Path(os.environ['FOURFACTORS_PROCESSED_DIR'])
    found = next((p for p in PROCESSED_CANDIDATES if p.exists()), None)
    assert found is not None, "Pipeline data/processed not found; set FOURFACTORS_PROCESSED_DIR"
    return found
