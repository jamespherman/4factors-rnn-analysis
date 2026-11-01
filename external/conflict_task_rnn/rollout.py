"""
Simulate the network on a batch of trials and sample saccade decisions.

Network dynamics do not depend on the agent's actions (no eye-position feedback),
so the full trial can be simulated first and the per-step policy sampled after.
The first non-fixate action ends the trial.
"""

import torch
import torch.nn.functional as F
from config import Config
from task import sample_trials, build_inputs, evaluate_choices, urgency


def valid_step_mask(cfg: Config, trials, T, device):
    """Steps at which a decision can be made: t <= go + deadline."""
    t = torch.arange(T, device=device, dtype=torch.float32) * cfg.dt
    return t[None, :] <= (trials["t_go"] + cfg.deadline)[:, None]


def sample_actions(logits, valid, generator=None, greedy=False):
    """Return act_step [B] (-1 if none), act_loc [B], actions [B,T]."""
    B, T, A = logits.shape
    probs = F.softmax(logits, -1)
    if greedy:
        a = probs.argmax(-1)
    else:
        a = torch.multinomial(probs.reshape(-1, A), 1, generator=generator).reshape(B, T)
    sac = (a > 0) & valid
    any_sac = sac.any(1)
    first = torch.where(any_sac, sac.float().argmax(1), torch.full((B,), -1, device=a.device))
    loc = torch.where(any_sac, a.gather(1, first.clamp(min=0)[:, None]).squeeze(1) - 1,
                      torch.full((B,), -1, device=a.device))
    return first.long(), loc.long(), a


def active_mask(act_step, valid):
    """Steps that belong to the trial (up to and including the saccade step)."""
    B, T = valid.shape
    steps = torch.arange(T, device=valid.device)[None, :]
    last = torch.where(act_step >= 0, act_step, valid.long().sum(1) - 1)
    return steps <= last[:, None]


@torch.no_grad()
def run_trials(net, cfg: Config, batch: int, device, generator=None, return_rates=False,
               noise=True, **trial_kw):
    trials = sample_trials(cfg, batch, device, generator=generator, **trial_kw)
    u, t = build_inputs(cfg, trials, device, generator=generator, noise=noise)
    R = net(u, noise=noise, generator=generator)
    urg = urgency(cfg, trials, t, generator=generator)
    logits = net.policy_logits(R, u, urg)
    valid = valid_step_mask(cfg, trials, u.shape[1], device)
    act_step, act_loc, _ = sample_actions(logits, valid, generator=generator)
    out = evaluate_choices(cfg, trials, act_step, act_loc)
    res = dict(trials=trials, out=out, act_step=act_step, act_loc=act_loc)
    if return_rates:
        res["rates"] = R
        res["u"] = u
        res["t"] = t
        res["logits"] = logits
    return res


def summarize(cfg: Config, res):
    """Scalar summary for logging."""
    tr, o = res["trials"], res["out"]
    c = o["completed"]
    s = dict(
        reward=o["reward"].mean().item(),
        completed=c.float().mean().item(),
        fixbreak=o["fixbreak"].float().mean().item(),
        early=o["early"].float().mean().item(),
        inaccurate=o["inaccurate"].float().mean().item(),
        noresp=o["noresp"].float().mean().item(),
    )
    conf = tr["conflict"]
    for name, m in (("conflict", conf), ("congruent", ~conf)):
        mm = m & c
        s[f"pbig_{name}"] = (o["chose_big"] & mm).float().sum().item() / max(mm.sum().item(), 1)
    s["rt_mean"] = o["rt"][c].mean().item() if c.any() else float("nan")
    s["rpt_mean"] = o["rpt"][c].mean().item() if c.any() else float("nan")
    by_dt = {}
    for d in cfg.delta_ts:
        m = c & conf & (tr["dT"] == d)
        by_dt[str(int(d))] = (o["chose_big"] & m).float().sum().item() / max(m.sum().item(), 1)
    s["pbig_conflict_by_dT"] = by_dt
    return s
