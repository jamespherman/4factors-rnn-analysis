"""
Hazard readout: sample saccades, score outcomes, and compute REINFORCE log-probabilities.

Outcome codes: 0 = completed (landed on a target), 1 = fixation break (event before go), 2 = no response by
go + response_window, 3 = inaccurate (saccade to an empty location on a single-target trial).
"""

import torch
import torch.nn.functional as F

COMPLETED, FIXBREAK, NORESP, INACCURATE = 0, 1, 2, 3


def sample_saccades(logits, valid, generator=None):
    """logits [B,T,2] (hazard per location), valid [B,T]. Returns event mask [B,T,2], step [B], loc [B]."""
    B, T, _ = logits.shape
    p = torch.sigmoid(logits)
    ev = (torch.rand(p.shape, device=p.device, generator=generator) < p) & valid[..., None]
    any_ev = ev.any(-1)                                   # [B,T]
    has = any_ev.any(1)
    step = torch.where(has, any_ev.float().argmax(1), torch.full((B,), -1, device=p.device))
    ar = torch.arange(B, device=p.device)
    at = ev[ar, step.clamp(min=0)]                        # [B,2] events at the first event step
    both = at.all(-1)
    pick_r = torch.rand(B, device=p.device, generator=generator) < 0.5
    loc = torch.where(at[:, 1] & (~at[:, 0] | pick_r), torch.ones_like(step), torch.zeros_like(step))
    loc = torch.where(has, loc, torch.full_like(loc, -1))
    return ev, step.long(), loc.long(), both


def log_prob(logits, ev, step, valid):
    """Sum of log-probabilities of the sampled per-location Bernoulli events up to and including `step`."""
    B, T, _ = logits.shape
    steps = torch.arange(T, device=logits.device)[None, :]
    last = torch.where(step >= 0, step, valid.long().sum(1) - 1)
    active = (steps <= last[:, None]) & valid
    lp = torch.where(ev, F.logsigmoid(logits), F.logsigmoid(-logits)).sum(-1)     # [B,T]
    return lp, active


def evaluate(cfg, inp, step, loc, generator=None):
    """Outcome, reward (model units), RT (ms, from go incl. efferent delay), chosen identity."""
    t_step = step.float() * cfg.dt            # outcome classification uses the event step itself
    t_ev = t_step.clone()                     # RT uses the (optionally sub-step / jittered) event time
    if getattr(cfg, "rt_substep", False):     # event time uniform within its step (continuous-time equivalent)
        t_ev = t_ev - cfg.dt * torch.rand(t_ev.shape, device=t_ev.device, generator=generator)
    if getattr(cfg, "rt_jitter_sd", 0.0) > 0:
        t_ev = t_ev + cfg.rt_jitter_sd * torch.randn(t_ev.shape, device=t_ev.device, generator=generator)
    t_go = inp["t_go"]
    responded = step >= 0
    fixbreak = responded & (t_step < t_go)
    valid_time = responded & ~fixbreak
    lc = loc.clamp(min=0)
    ar = torch.arange(loc.shape[0], device=loc.device)
    on_target = valid_time & (inp["present"][ar, lc] > 0)
    inaccurate = valid_time & ~on_target
    outcome = torch.full_like(step, NORESP)
    outcome = torch.where(on_target, torch.full_like(step, COMPLETED), outcome)
    outcome = torch.where(fixbreak, torch.full_like(step, FIXBREAK), outcome)
    outcome = torch.where(inaccurate, torch.full_like(step, INACCURATE), outcome)
    reward = torch.where(on_target, inp["rew"][ar, lc], torch.full_like(t_ev, cfg.reward_abort))
    chosen_id = torch.where(on_target, inp["ids"][ar, lc], torch.zeros_like(t_ev))
    rt = torch.where(valid_time, t_ev + cfg.efferent_delay - t_go, torch.full_like(t_ev, float("nan")))
    return dict(outcome=outcome, reward=reward, choice_loc=torch.where(on_target, loc, -torch.ones_like(loc)),
                chosen_id=chosen_id, rt=rt)


def valid_mask(cfg, inp, T):
    t = inp["t"]
    return t[None, :] <= (inp["t_go"] + cfg.response_window)[:, None]
