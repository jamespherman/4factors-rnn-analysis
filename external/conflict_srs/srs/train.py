"""
Train an srsClassic hub/resolver network with REINFORCE (learned value baseline) and a 3-stage curriculum:
  S1  single-target trials only, flat reward: learn to hold fixation (SNr + fixation drive) and saccade after go
  S2  50/50 single/choice trials with real block values; belief = block means
  S3  as S2 with noisy beliefs (belief_noise ms SD around the block means)

Usage (GPU machine, e.g. GCP):
  python -m srs.train --arch B --seed 0 --name B_s0 [--out results/srs] [--set n_up=64 commissural=false ...]
Outputs: <out>/<name>/{config.json, log.json, model.pt}; model.pt holds state_dict + config (load with load_net()).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .config import SRSConfig
from .network import SRSNet
from .task import make_training_trials, build_inputs
from .rollout import sample_saccades, log_prob, evaluate, valid_mask

ROOT = Path(__file__).resolve().parents[2]


def parse_set(cfg, items):
    for it in items or []:
        k, v = it.split("=", 1)
        cur = getattr(cfg, k)
        if isinstance(cur, bool):
            val = v.lower() in ("1", "true", "yes")
        elif isinstance(cur, tuple):
            val = tuple(type(cur[0])(x) for x in v.split(","))
        else:
            val = type(cur)(v)
        setattr(cfg, k, val)
    return cfg


def load_net(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    cfg = SRSConfig.from_dict(ck["config"])
    cfg.device = device
    net = SRSNet(cfg).to(device)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net


def step_batch(net, cfg, table, gen):
    dev = next(net.parameters()).device
    inp = build_inputs(cfg, table, dev, generator=gen)
    out = net(inp, generator=gen)
    logits = net.hazard_logits(out["pool"])
    T = logits.shape[1]
    valid = valid_mask(cfg, inp, T)
    ev, step, loc, _ = sample_saccades(logits.detach(), valid, generator=gen)
    res = evaluate(cfg, inp, step, loc, generator=gen)
    lp, active = log_prob(logits, ev, step, valid)
    tfrac = (inp["t"] / inp["t"][-1])[None, :, None].expand(logits.shape[0], T, 1)
    V = net.value(out, tfrac)
    R = res["reward"]
    if cfg.time_cost > 0:
        R = R - cfg.time_cost * torch.nan_to_num(res["rt"], nan=0.0).clamp(min=0) * (res["outcome"] == 0).float()
    act = active.float()
    n = act.sum()
    adv = (R[:, None] - V).detach()
    loss_pi = -(lp * adv * act).sum() / n
    loss_v = (((R[:, None] - V) ** 2) * act).sum() / n
    loss_rate = ((out["sc"] ** 2).mean(-1) * act).sum() / n
    loss = loss_pi + cfg.value_coef * loss_v + cfg.rate_reg * loss_rate + cfg.weight_reg * (net.W_sc ** 2).sum()
    return loss, res


def summarize(res, table):
    o = res["outcome"].cpu().numpy()
    loc = res["choice_loc"].cpu().numpy()
    rt = res["rt"].cpu().numpy()
    ch = (table["trial_type"] == 1) & (o == 0)
    s = dict(reward=float(res["reward"].mean()), completed=float((o == 0).mean()), fixbreak=float((o == 1).mean()),
             noresp=float((o == 2).mean()), inaccurate=float((o == 3).mean()),
             rt=float(np.nanmedian(rt[o == 0])) if (o == 0).any() else float("nan"))
    if ch.any():
        rich = (table["rew_R"] > table["rew_L"]).astype(int)
        s["p_high"] = float((loc[ch] == rich[ch]).mean())
        s["p_right"] = float((loc[ch] == 1).mean())
    return s


def train(cfg, name, out_root=None, verbose=True):
    out_dir = Path(out_root or ROOT / "results" / "srs") / name
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    net = SRSNet(cfg).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    gen = torch.Generator(device=dev)
    gen.manual_seed(cfg.seed + 1)
    json.dump(cfg.to_dict(), open(out_dir / "config.json", "w"), indent=1)
    log, t0, it_total = [], time.time(), 0
    for stage, n_it in zip(("S1", "S2", "S3"), cfg.stage_iters):
        for it in range(1, n_it + 1):
            it_total += 1
            table = make_training_trials(cfg, cfg.batch_size, rng, stage)
            if stage in cfg.nuisance_stages:
                n = cfg.batch_size
                w = rng.uniform(cfg.train_engaged_min, 1.0)
                table["engaged"] = (rng.random(n) < w).astype(float)
                table["side_bias"] = rng.uniform(-cfg.train_side_bias_max, cfg.train_side_bias_max, n)
            loss, res = step_batch(net, cfg, table, gen)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            opt.step()
            if it % cfg.eval_every == 0 or it == 1 or it == n_it:
                s = summarize(res, table)
                s.update(stage=stage, iter=it_total, loss=float(loss), elapsed=time.time() - t0)
                log.append(s)
                if verbose:
                    print(f"[{name}] {stage} it {it_total:5d} rew {s['reward']:.3f} comp {s['completed']:.2f} "
                          f"fb {s['fixbreak']:.2f} nr {s['noresp']:.2f} inacc {s['inaccurate']:.2f} "
                          f"Phigh {s.get('p_high', float('nan')):.2f} R {s.get('p_right', float('nan')):.2f} "
                          f"RT {s['rt']:.0f} | {s['elapsed']:.0f}s", flush=True)
                json.dump(log, open(out_dir / "log.json", "w"), indent=1)
                torch.save({"state_dict": net.state_dict(), "config": cfg.to_dict(), "iter": it_total,
                            "stage": stage}, out_dir / "model.pt")
    return net, log


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="B", choices=["A", "B", "C"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--set", nargs="*", default=[])
    a = p.parse_args()
    cfg = parse_set(SRSConfig(arch=a.arch, seed=a.seed), a.set)
    train(cfg, a.name, a.out)
