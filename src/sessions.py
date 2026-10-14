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

# Original three-session set (short ids '08_13' etc. are Newton sessions)
DEFAULT_SESSIONS = ['08_13', '08_14', '08_15']

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


def export_path(session: str) -> Path:
    """Export file of a session in data/."""
    return DATA_DIR / f'rnn_export_{session_name(session)}.mat'


def available_sessions() -> List[str]:
    """Full names of the sessions with an export in data/."""
    return sorted(p.stem.replace('rnn_export_', '') for p in DATA_DIR.glob('rnn_export_*.mat'))


def outcome_field(data: dict):
    """Name of the export's per-trial outcome field, or None."""
    return next((f for f in OUTCOME_FIELDS if f in data), None)


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


def load_export(session: str, trials: str = 'all', path=None) -> dict:
    """
    Load a session's export restricted to a trial set.

    Args:
        session: Session name (short forms accepted)
        trials: 'all' or 'verified'
        path: Export file to load instead of data/rnn_export_<name>.mat
    """
    path = Path(path) if path is not None else export_path(session)
    assert path.exists(), f"Missing export {path}"
    return select_trial_set(load_mat_file(str(path)), session, trials)


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
