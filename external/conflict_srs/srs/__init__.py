"""srsClassic hub/resolver bilateral SC models (plan v2, 2026-10-31). See README.md."""
from .config import SRSConfig
from .network import SRSNet, SCIndex
from .handbuilt import build_handbuilt, HB_DEFAULTS
from .simulate import simulate, apply_perturbation, reset_perturbation, behavior_summary
from .task import make_session, make_training_trials, belief_delta_rule, as_table
