"""
Configuration for the bilateral SC E-I RNN trained on the conflict task.

All times are in ms. Conventions follow the Aim 1 data-fitting RNN
(4factors-rnn-analysis): softplus nonlinearity, tau = 50 ms, noise_scale = 0.1
(state noise scaled by sqrt(dt/tau)), Dale's law via |W_raw| * sign mask.
dt is 10 ms here (vs. 25 ms bins in Aim 1) because the tachometric curve
needs finer temporal resolution than 25 ms.

Delta-t convention (matches the task code): dt_stim = target onset - go.
Positive values = targets appear AFTER the go signal (short processing time).
Processing time (rPT) = saccade onset - target onset = RT(from go) - dt_stim.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Tuple


@dataclass
class Config:
    # ---------------- Network ----------------
    n_exc_hemi: int = 80          # E units per hemisphere
    n_inh_hemi: int = 20          # I units per hemisphere
    n_loc_hemi: int = 4           # target locations per hemifield (topographic E pools)
    frac_inh_commissural: float = 0.5   # fraction of I units allowed to project commissurally
    p_within: float = 0.8         # within-hemisphere connection probability
    p_commissural: float = 0.3    # commissural (I -> contralateral E/I) connection probability
    commissural: bool = True      # False = train without any cross-hemispheric connections
    comm_exc_to_inh: bool = False # control: also allow E -> contralateral I (cross-midline feedforward inhibition)
    p_comm_exc: float = 0.1       # connection probability for that E -> contralateral I pathway
    tau: float = 50.0
    dt: float = 10.0
    noise_scale: float = 0.1      # state noise (same convention as Aim 1 RNN)
    input_noise: float = 0.05     # additive noise on input channels
    spectral_radius: float = 0.9
    nonlinearity: str = "softplus"

    # ---------------- Inputs ----------------
    learn_sensory_weights: bool = False  # False: visual/goal input weights are fixed (hard-wired)
    sensory_to_inh: bool = False  # if True, visual input also drives I units of the hemisphere
    goal_to_inh: bool = False     # if True, goal input also drives I units of the hemisphere
    vis_gain: float = 1.0
    goal_gain: float = 1.0
    reward_context_input: bool = False  # tonic hemifield reward-context input (learnable weights)

    # visual (stimulus-driven) signal: onset at target_on + vis_latency,
    # transient that decays to a sustained level. Salience sets the transient peak.
    vis_latency: float = 50.0
    vis_tau_decay: float = 50.0
    sal_peak_high: float = 1.5
    sal_peak_low: float = 0.5
    vis_sustained: float = 0.3
    # goal-directed (reward) signal: onset at target_on + goal_latency, sustained,
    # amplitude encodes the reward value of the target's hemifield.
    goal_latency: float = 50.0     # same as visual latency: switch arises from salience transient decay
    goal_rise_tau: float = 20.0
    goal_big: float = 1.0
    goal_small: float = 0.2

    # ---------------- Task timing ----------------
    fix_min: float = 300.0        # fixation duration before go (model; monkey: 1.0-1.2 s)
    fix_max: float = 500.0
    delta_ts: Tuple[float, ...] = (-150., -100., -50., 0., 50., 100., 150.)
    deadline: float = 450.0       # response window after go (current task code)
    go_latency: float = 50.0      # afferent latency of the fixation-offset (go) signal
    efferent_delay: float = 20.0  # policy commit -> saccade onset (added in analysis only)

    # ---------------- Urgency / saccade generator ----------------
    # fix_readout: "network" = fixation logit read out from E units (learned);
    #              "input"   = fixation logit driven only by the fixation-point input
    #                          (saccade generator gate; the SC network controls where,
    #                          and can delay only by keeping its E pools quiet).
    fix_readout: str = "input"
    # go-locked urgency: after go (+go_latency) the fixation logit falls linearly with
    # slope urgency_slope (logit units / 100 ms); slope varies across trials (lognormal sd).
    urgency_slope: float = 6.0
    urgency_sd: float = 0.3
    urgency_delay: float = 50.0   # extra delay after go + go_latency before the ramp starts

    # ---------------- Rewards ----------------
    reward_big: float = 1.0
    reward_ratio: float = 1.5     # big / small (current task: 1.5; pre-Apr-29: 2.0)
    abort_reward: float = 0.0     # fixation break, no response, inaccurate
    time_cost: float = 0.0        # optional per-ms urgency cost after go (0 = none)

    # ---------------- Training ----------------
    batch_size: int = 256
    n_iters: int = 6000
    lr: float = 1e-3
    grad_clip: float = 1.0
    entropy_coef: float = 1e-3
    value_coef: float = 0.5
    rate_reg: float = 1e-3
    weight_reg: float = 1e-5
    fix_bias_init: float = 4.0
    seed: int = 0
    device: str = "cuda"
    eval_every: int = 500
    eval_batch: int = 4096

    @property
    def reward_small(self) -> float:
        return self.reward_big / self.reward_ratio

    @property
    def n_hemi(self) -> int:
        return self.n_exc_hemi + self.n_inh_hemi

    @property
    def n_total(self) -> int:
        return 2 * self.n_hemi

    @property
    def n_loc(self) -> int:
        return 2 * self.n_loc_hemi

    def to_dict(self):
        d = asdict(self)
        d["delta_ts"] = list(self.delta_ts)
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        if "delta_ts" in d:
            d["delta_ts"] = tuple(d["delta_ts"])
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)
