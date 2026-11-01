"""
Conflict task (Herman lab) as a trial generator for the RNN.

Trial structure (per trial):
  t = 0            fixation point on (fixation input = 1)
  t = t_go         go signal: fixation point off (fixation input -> 0)
  t = t_go + dT    targets on: one target in each hemifield, each at one of
                   n_loc_hemi positions. One is high-salience, the other low.
                   dT in cfg.delta_ts (dT = target onset - go).
  t = t_go + deadline   response window closes.

Reward is mapped to hemifield (big side). Conflict trial: high-salience target
on the small-reward side. Congruent: high-salience target on the big side.
Big side and salient side are drawn independently -> 50% conflict.

Location indexing: 0..n_loc_hemi-1 = LEFT hemifield, n_loc_hemi..2*n_loc_hemi-1 = RIGHT.
The RIGHT SC represents the LEFT hemifield and vice versa (see network.py).

Input channels:
  0                      fixation
  1 .. n_loc             visual (stimulus-driven; salience-dependent transient)
  1+n_loc .. 2*n_loc     goal (reward-value signal, delayed)
  1+2*n_loc, 2+2*n_loc   reward context (tonic): [left big, right big]
"""

import torch
from config import Config

LEFT, RIGHT = 0, 1


def n_inputs(cfg: Config) -> int:
    return 1 + 2 * cfg.n_loc + 2


def sample_trials(cfg: Config, batch: int, device, generator=None,
                  delta_ts=None, conflict=None, big_side=None, sal_side=None, single_target=False):
    """Draw trial parameters. Optional overrides fix a variable for all trials.
    single_target=True: probe trials with one target only (as in Phase 1 of the monkey task);
    the present side is random, its salience is random (high/low), reward follows the block (big_side)."""
    g = generator
    dts = torch.tensor(delta_ts if delta_ts is not None else cfg.delta_ts, device=device)
    nfix = int(round((cfg.fix_max - cfg.fix_min) / cfg.dt)) + 1
    t_go = cfg.fix_min + cfg.dt * torch.randint(0, nfix, (batch,), device=device, generator=g)
    d = dts[torch.randint(0, len(dts), (batch,), device=device, generator=g)]
    if big_side is None:
        big = torch.randint(0, 2, (batch,), device=device, generator=g)
    else:
        big = torch.full((batch,), big_side, device=device, dtype=torch.long)
    if conflict is not None:
        sal = big if not conflict else 1 - big
    elif sal_side is not None:
        sal = torch.full((batch,), sal_side, device=device, dtype=torch.long)
    else:
        sal = torch.randint(0, 2, (batch,), device=device, generator=g)
    pos_l = torch.randint(0, cfg.n_loc_hemi, (batch,), device=device, generator=g)
    pos_r = torch.randint(0, cfg.n_loc_hemi, (batch,), device=device, generator=g)
    tr = dict(t_go=t_go, dT=d, big_side=big, sal_side=sal,
              loc_left=pos_l, loc_right=pos_r + cfg.n_loc_hemi,
              conflict=(big != sal))
    if single_target:
        present = torch.randint(0, 2, (batch,), device=device, generator=g)
        hi = torch.randint(0, 2, (batch,), device=device, generator=g).bool()
        # for single-target trials sal_side marks the present target's side if it is high salience,
        # else the absent side (so "present target is salient" == (sal_side == present))
        tr["sal_side"] = torch.where(hi, present, 1 - present)
        tr["single"] = present
        tr["conflict"] = torch.zeros(batch, dtype=torch.bool, device=device)
    return tr


def n_steps(cfg: Config) -> int:
    return int(round((cfg.fix_max + cfg.deadline) / cfg.dt)) + 1


def build_inputs(cfg: Config, trials: dict, device, generator=None, noise=True):
    """Return inputs [batch, T, n_in] and time axis (ms) [T]."""
    B = trials["t_go"].shape[0]
    T = n_steps(cfg)
    t = torch.arange(T, device=device, dtype=torch.float32) * cfg.dt  # [T]
    t_go = trials["t_go"][:, None]
    t_on = (trials["t_go"] + trials["dT"])[:, None]
    u = torch.zeros(B, T, n_inputs(cfg), device=device)

    # fixation input: on until go (+ afferent latency: fixation offset is itself a visual event)
    u[:, :, 0] = (t[None, :] < t_go + cfg.go_latency).float()

    # visual transient: salience-dependent peak decaying to sustained level
    tv = t[None, :] - (t_on + cfg.vis_latency)
    on_v = (tv >= 0).float()
    decay = torch.exp(-tv.clamp(min=0) / cfg.vis_tau_decay)
    # goal signal: delayed, rises with time constant goal_rise_tau, sustained
    tg = t[None, :] - (t_on + cfg.goal_latency)
    on_g = (tg >= 0).float()
    rise = 1 - torch.exp(-tg.clamp(min=0) / cfg.goal_rise_tau)

    ar = torch.arange(B, device=device)
    for side, loc in ((LEFT, trials["loc_left"]), (RIGHT, trials["loc_right"])):
        is_sal = (trials["sal_side"] == side).float()[:, None]
        peak = cfg.sal_peak_low + (cfg.sal_peak_high - cfg.sal_peak_low) * is_sal
        vis = on_v * (cfg.vis_sustained + (peak - cfg.vis_sustained) * decay)
        is_big = (trials["big_side"] == side).float()[:, None]
        gamp = cfg.goal_small + (cfg.goal_big - cfg.goal_small) * is_big
        goal = on_g * rise * gamp
        if "single" in trials:
            pres = (trials["single"] == side).float()[:, None]
            vis, goal = vis * pres, goal * pres
        u[ar, :, 1 + loc] = vis
        u[ar, :, 1 + cfg.n_loc + loc] = goal

    # reward context (tonic, whole trial): which hemifield is big in this block
    u[ar, :, 1 + 2 * cfg.n_loc + trials["big_side"]] = 1.0

    if noise and cfg.input_noise > 0:
        u = u + cfg.input_noise * torch.randn(u.shape, device=device, generator=generator)
    return u, t


def evaluate_choices(cfg: Config, trials: dict, act_step: torch.Tensor, act_loc: torch.Tensor):
    """
    Compute outcome of each trial from the (first) saccade.
    act_step: step index of saccade commit (-1 if none within deadline)
    act_loc:  location chosen (0..n_loc-1), undefined when act_step == -1
    Returns dict with reward and outcome categories.
    """
    t_act = act_step.float() * cfg.dt
    t_go = trials["t_go"]
    t_on = t_go + trials["dT"]
    responded = act_step >= 0
    fixbreak = responded & (t_act < t_go)
    early = responded & ~fixbreak & (t_act < t_on)          # after go but before targets: nothing to land on
    valid_time = responded & ~fixbreak & ~early
    chose_left = act_loc == trials["loc_left"]
    chose_right = act_loc == trials["loc_right"]
    if "single" in trials:
        chose_left = chose_left & (trials["single"] == LEFT)
        chose_right = chose_right & (trials["single"] == RIGHT)
    on_target = valid_time & (chose_left | chose_right)
    inaccurate = valid_time & ~on_target
    chosen_side = torch.where(chose_right, torch.ones_like(act_loc), torch.zeros_like(act_loc))
    chose_big = on_target & (chosen_side == trials["big_side"])
    chose_sal = on_target & (chosen_side == trials["sal_side"])
    reward = torch.full_like(t_act, cfg.abort_reward)
    reward = torch.where(on_target & chose_big, torch.full_like(t_act, cfg.reward_big), reward)
    reward = torch.where(on_target & ~chose_big, torch.full_like(t_act, cfg.reward_small), reward)
    if cfg.time_cost > 0:
        reward = reward - cfg.time_cost * (t_act - t_go).clamp(min=0) * on_target.float()
    rt = t_act + cfg.efferent_delay - t_go       # RT from go (saccade onset)
    rpt = t_act + cfg.efferent_delay - t_on      # raw processing time
    return dict(reward=reward, completed=on_target, fixbreak=fixbreak, early=early,
                inaccurate=inaccurate, noresp=~responded, chose_big=chose_big,
                chose_sal=chose_sal, chosen_side=chosen_side, rt=rt, rpt=rpt)


def urgency(cfg: Config, trials: dict, t: torch.Tensor, generator=None):
    """Go-locked urgency ramp [B, T] subtracted from the fixation logit."""
    B = trials["t_go"].shape[0]
    if cfg.urgency_slope <= 0:
        return torch.zeros(B, t.shape[0], device=t.device)
    if "urg_k" not in trials:
        z = torch.randn(B, device=t.device, generator=generator)
        trials["urg_k"] = cfg.urgency_slope * torch.exp(cfg.urgency_sd * z - 0.5 * cfg.urgency_sd ** 2)
    start = trials["t_go"] + cfg.go_latency + cfg.urgency_delay
    return trials["urg_k"][:, None] * (t[None, :] - start[:, None]).clamp(min=0) / 100.0
