"""
simulate(): run a trained or hand-built network on a trial table and return
  table_out  dict of arrays: the input table plus outcome, choice_loc (0 L / 1 R / -1 none), chosen_id, reward_ms,
             rt (ms from go), go_step, onset_step_L/R
  rates      dict: 'sc' [n_trials, T, N_sc] float32, 'snr' [n, T, 2], 'dis' [n, T, 2], 'up' [n, T, n_up] (optional),
             'pool' [n, T, 2]; time axis 't' (ms from trial start; fixation onset = 0)
  units      dict of arrays describing SC units: sc_side ('L'/'R' colliculus), type (E_pool/E_other/I_local/I_comm),
             field (hemifield represented), is_exc, pool_loc (0/1 for pool units, -1 otherwise)
Rates are model rates (softplus units, arbitrary scale); Poisson conversion and matched sampling are done downstream.

Perturbations (apply_perturbation / perturb= argument), all reversible via reset_perturbation:
  {"bicuculline": {"sc": "L"|"R", "d": 0.3, "scope": "side"|"pool", "snr": True}}
      scales GABA-A synapses onto E units at the site by (1-d): local and commissural I -> E weights, and (if snr)
      SNr -> E inhibition. scope "pool" restricts the site to the E pool.
  {"muscimol": {"sc": "L"|"R", "d": 1.0}}       scales the output of all units of that SC by (1-d)
  {"comm_cut": True}                            removes cross-SC weights
  {"silence": {"sc": "L"|"R"|"both", "types": ["I_comm", ...]}}
  {"snr_value_clamp": True}                     replaces the upstream disinhibition by its trial-average time course
                                                (value-independent upstream signal; upstream manipulation test)
"""

import numpy as np
import torch

from .task import as_table, build_inputs
from .rollout import sample_saccades, evaluate, valid_mask

TYPE_INDEX = {"E_pool": 0, "E_other": 1, "I_local": 2, "I_comm": 3}


def reset_perturbation(net):
    net.w_gain.fill_(1.0)
    net.snr_gain.fill_(1.0)
    net.unit_gain.fill_(1.0)
    net.dis_override = None


def _sc_units(net, sc):
    side = {"L": 0, "R": 1}
    idx = net.idx
    if sc == "both":
        return np.ones(idx.n, bool)
    return idx.side == side[sc]


def apply_perturbation(net, perturb):
    reset_perturbation(net)
    if not perturb:
        return
    idx = net.idx
    dev = net.w_gain.device
    for kind, p in perturb.items():
        if kind == "bicuculline":
            site = _sc_units(net, p["sc"]) & idx.is_exc
            if p.get("scope", "side") == "pool":
                site &= idx.is_pool
            post = torch.tensor(site, device=dev)
            pre_i = torch.tensor(idx.is_inh, device=dev)
            net.w_gain[post[:, None] & pre_i[None, :]] *= (1 - p["d"])
            if p.get("snr", True):
                net.snr_gain[post] *= (1 - p["d"])
        elif kind == "muscimol":
            net.unit_gain[torch.tensor(_sc_units(net, p["sc"]), device=dev)] *= (1 - p.get("d", 1.0))
        elif kind == "comm_cut" and p:
            net.w_gain *= (1 - net.cross_mask)
        elif kind == "silence":
            m = _sc_units(net, p.get("sc", "both")) & np.isin(idx.type, [TYPE_INDEX[t] for t in p["types"]])
            net.unit_gain[torch.tensor(m, device=dev)] = 0.0
        elif kind == "snr_value_clamp" and p:
            net.dis_override = lambda dis, inp: dis.mean(0, keepdim=True).expand_as(dis)
        else:
            raise ValueError(kind)


@torch.no_grad()
def simulate(net, table, seed=0, perturb=None, chunk=512, return_rates=True, return_up=False, noise=True):
    cfg = net.cfg
    dev = next(net.parameters()).device
    table = as_table(table)
    n = len(table["trial_type"])
    apply_perturbation(net, perturb)
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    keys = ["outcome", "choice_loc", "chosen_id", "reward", "rt"]
    res = {k: [] for k in keys}
    rates = {"sc": [], "snr": [], "dis": [], "pool": []}
    if return_up:
        rates["up"] = []
    T_all = None
    from .task import n_steps
    T_all = n_steps(cfg, table)                     # common length for all chunks
    for s in range(0, n, chunk):
        sub = {k: v[s:s + chunk] for k, v in table.items()}
        inp = build_inputs(cfg, sub, dev, generator=g)
        T = inp["t"].shape[0]
        out = net(inp, noise=noise, generator=g)
        logits = net.hazard_logits(out["pool"], inp)
        valid = valid_mask(cfg, inp, T)
        _, step, loc, _ = sample_saccades(logits, valid, generator=g)
        ev = evaluate(cfg, inp, step, loc, generator=g)
        for k in keys:
            res[k].append(ev[k].cpu().numpy())
        if return_rates:
            for k in rates:
                a = out[k].float().cpu().numpy()
                pad = T_all - a.shape[1]
                if pad > 0:
                    a = np.concatenate([a, np.repeat(a[:, -1:], pad, 1)], 1)
                rates[k].append(a.astype(np.float32))
    reset_perturbation(net)
    out_t = dict(table)
    for k in keys:
        out_t[k] = np.concatenate(res[k])
    out_t["reward_ms"] = out_t.pop("reward") * cfg.value_scale
    out_t["go_step"] = np.round((table["fix_dur"] + table["delay"]) / cfg.dt).astype(int)
    out_t["onset_step_L"] = np.round((table["fix_dur"] + table["onset_L"]) / cfg.dt).astype(int)
    out_t["onset_step_R"] = np.round((table["fix_dur"] + table["onset_R"]) / cfg.dt).astype(int)
    result = dict(table=out_t, units=net.idx.meta(), t=np.arange(T_all) * cfg.dt)
    if return_rates:
        result["rates"] = {k: np.concatenate(v) for k, v in rates.items()}
    return result


def behavior_summary(res):
    """Quick behavioural summary (choice trials): P(chose richer per belief / per reward), side bias, RT."""
    t = res["table"]
    comp = t["outcome"] == 0
    ch = comp & (t["trial_type"] == 1)
    out = dict(n=len(comp), completed=comp.mean(), fixbreak=(t["outcome"] == 1).mean(),
               noresp=(t["outcome"] == 2).mean(), inaccurate=(t["outcome"] == 3).mean())
    if ch.any():
        rich_loc = (t["rew_R"] > t["rew_L"]).astype(int)
        out["p_chose_higher_reward"] = (t["choice_loc"][ch] == rich_loc[ch]).mean()
        out["p_right"] = (t["choice_loc"][ch] == 1).mean()
        out["rt_choice"] = np.nanmedian(t["rt"][ch])
    st = comp & (t["trial_type"] == 0)
    if st.any():
        out["rt_single"] = np.nanmedian(t["rt"][st])
    return out
