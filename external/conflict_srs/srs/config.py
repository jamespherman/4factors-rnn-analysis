"""
Configuration for the srsClassic hub/resolver models (plan v2, 2026-10-31).

All times in ms. Locations: 0 = LEFT hemifield, 1 = RIGHT hemifield. SC side 0 = LEFT SC, which represents the
RIGHT hemifield (location 1); SC side 1 = RIGHT SC, representing location 0.
Identities: 1 = T1 (horizontal bar), 2 = T2 (vertical bar), 0 = no target.
Rewards and beliefs are in ms of juice (as in the task code) and are divided by `value_scale` inside the model.
"""

from dataclasses import dataclass, asdict


@dataclass
class SRSConfig:
    # ---------------- architecture ----------------
    arch: str = "B"                 # "B" priority hub (graded), "C" outcome hub (upstream WTA), "A" resolver
    n_pool: int = 40                # E units in the pool at the SC's contralateral target location
    n_e_other: int = 40             # non-pool E units per SC
    n_i_local: int = 10             # local-only I units per SC
    n_i_comm: int = 10              # I units that may also project to the other SC
    commissural: bool = True
    p_within: float = 0.8
    p_comm: float = 0.3
    n_up: int = 64                  # upstream (BG-like) module units
    tau: float = 50.0
    tau_up: float = 50.0
    dt: float = 10.0
    noise_sc: float = 0.1           # SC state noise (same convention as the Aim 1 / conflict-task RNNs)
    noise_up: float = 0.1           # upstream state noise
    slow_tau: float = 200.0         # slow (OU) noise inputs onto the E pools
    slow_shared: float = 0.0        # amplitude of the slow noise shared by both pools
    slow_side: float = 0.0          # amplitude of independent slow noise per pool
    spectral_radius: float = 0.9

    # ---------------- inputs ----------------
    vis_latency: float = 50.0       # visual / identity onset latency into SC
    up_latency: float = 70.0        # identity onset latency into the upstream module
    go_latency: float = 50.0        # fixation-offset (go) latency
    vis_sustained: float = 0.5
    vis_transient: float = 0.5
    vis_tau: float = 50.0
    sal_gain: float = 0.1           # weak: isoluminant salience (set ~1 for the luminant redesign)
    w_vis: float = 1.0              # fixed visual weight onto pool E
    value_scale: float = 300.0      # ms of juice -> model units
    a_input_gain: float = 1.0       # A: fixed random identity/belief weights onto SC (scale)
    a_input_targets: str = "all"    # A: which SC units get identity/belief input: "all" | "E" | "E_other"
    bias_locus: str = "SC"          # "SC" (side bias onto pools) or "upstream"
    learn_sc_fix: bool = True       # learnable fixation-input weights onto SC units

    # ---------------- SNr (inhibitory upstream output) ----------------
    snr_tonic: float = 1.0          # tonic SNr activity (fixed)
    snr_fix: float = 2.0            # extra SNr drive while the fixation point is on (fixed)
    w_snr: float = 1.0              # fixed SNr -> pool E inhibitory weight
    dis_max: float = 2.0            # maximal disinhibition the upstream can deliver per location
    c_beta: float = 5.0             # C: softmax gain of the upstream WTA (over L, R and a "no-go" option)
    c_dis_gain: float = 1.0         # C: scales the categorical disinhibition (RT knob; lower = slower)
    dis_tau: float = 0.0            # optional low-pass (ms) on the upstream disinhibition, all archs (0 = off)

    # ---------------- readout (stochastic hazard per pool) ----------------
    hazard_gain_init: float = 10.0
    hazard_theta_init: float = 1.0
    efferent_delay: float = 20.0
    rt_substep: bool = True         # saccade time uniform within the event step (sub-dt resolution)
    rt_jitter_sd: float = 0.0       # extra Gaussian motor jitter on saccade time (ms)
    response_window: float = 450.0

    # ---------------- task ----------------
    fix_min: float = 500.0
    fix_max: float = 800.0
    delay: float = 450.0            # target onset -> go (recorded task: fixed 450 ms)
    reward_abort: float = 0.0

    # ---------------- training ----------------
    batch_size: int = 256
    lr: float = 1e-3
    grad_clip: float = 1.0
    rate_reg: float = 1e-3
    weight_reg: float = 1e-5
    value_coef: float = 0.5
    stage_iters: tuple = (1500, 1500, 1500)   # S1 single-target, S2 true values, S3 noisy beliefs
    belief_noise: float = 40.0      # ms SD of belief around block means in training stage S3
    p_choice: float = 0.5           # fraction of choice trials in training stages S2-S3
    seed: int = 0
    device: str = "cuda"
    eval_every: int = 250

    def to_dict(self):
        d = asdict(self)
        d["stage_iters"] = list(self.stage_iters)
        return d

    @classmethod
    def from_dict(cls, d):
        d = {k: v for k, v in dict(d).items() if k in cls.__dataclass_fields__}
        if "stage_iters" in d:
            d["stage_iters"] = tuple(d["stage_iters"])
        return cls(**d)

    @property
    def n_side(self):
        return self.n_pool + self.n_e_other + self.n_i_local + self.n_i_comm

    @property
    def n_sc(self):
        return 2 * self.n_side
