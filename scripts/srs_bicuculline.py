#!/usr/bin/env python3
"""
In-silico unilateral SC bicuculline (plans/2026-10-31_bicuculline_plan.md) for hand-built and trained v2 resolver (A,
AE), priority hub (B) and outcome hub (C) networks (conflict-task-RNN code/srs, snapshot in external/conflict_srs).

Bicuculline at the LEFT SC (represents the RIGHT field, location 1) scales GABA-A synapses onto its E units by (1-d):
'full' = local/commissural I -> E and SNr -> E; 'local' = I -> E only; 'snr' = SNr -> E only.
Per network and condition, pooled over seeds and trial tables (08-20 recorded sequence with fitted beliefs + synthetic
sessions): fixation breaks (rate, direction), single-target RT and RT-value dependence per field, choice bias and value
weights (logistic), delay-window pool activity vs value per field. Writes results/hub_models/bicuculline/.
Usage: python scripts/srs_bicuculline.py [hand|trained|all|name,name,...] [n_seeds]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'external' / 'conflict_srs'))
from srs import SRSConfig, build_handbuilt  # noqa: E402
from srs.simulate import apply_perturbation, reset_perturbation  # noqa: E402
from srs.task import from_recorded, make_session, build_inputs  # noqa: E402
from srs.rollout import sample_saccades, evaluate, valid_mask  # noqa: E402
sys.path.insert(0, str(ROOT)); from src.srs_load import load_net  # noqa: E402

torch.set_num_threads(1)
OUT = ROOT / 'results' / 'hub_models' / 'bicuculline'
BELIEF = ROOT / 'results' / 'hub_models' / 'belief' / '08-20_trials_with_belief.csv'
TRAINED = ROOT / 'results' / 'hub_models' / 'trained_v2'
HAND = {'A': {}, 'B': {}, 'C': {'c_gain': 1.1, 'theta': 0.75}}
TRAINED_NETS = [f'{a}_s{s}' for a in ('A', 'AE', 'B') for s in range(6)] + ['C_s0', 'C_s4', 'C_s5']
# d = 0.9 drives every network into irrepressible saccades to the injected field (all trials fixation breaks), so the
# dose series stays low; 'local'/'snr' dissociate the two GABA-A sources at a matched dose
CONDS = {'ctrl': None, 'full05': ('full', 0.05), 'full10': ('full', 0.1), 'full20': ('full', 0.2),
         'full35': ('full', 0.35), 'local20': ('local', 0.2), 'snr20': ('snr', 0.2), 'snr50': ('snr', 0.5),
         'clamp': ('clamp', 0), 'musc50': ('musc', 0.5)}
SHAPE_VIS = 0.25


def perturb(net, kind, d):
    """Bicuculline at the LEFT SC's E units (RIGHT field)."""
    reset_perturbation(net)
    if kind == 'full':
        apply_perturbation(net, {'bicuculline': {'sc': 'L', 'd': d, 'snr': True}})
    elif kind == 'local':
        apply_perturbation(net, {'bicuculline': {'sc': 'L', 'd': d, 'snr': False}})
    elif kind == 'clamp':        # upstream disinhibition replaced by its trial average (value-free upstream)
        apply_perturbation(net, {'snr_value_clamp': True})
    elif kind == 'musc':
        apply_perturbation(net, {'muscimol': {'sc': 'L', 'd': d}})
    elif kind == 'snr':
        site = torch.tensor((net.idx.side == 0) & net.idx.is_exc)
        net.snr_gain[site] *= (1 - d)


@torch.no_grad()
def run(net, table, seed):
    """Simulate; return per-trial outcome, saccade loc (incl. fixation breaks), rt, delay-window pool rates."""
    cfg = net.cfg
    g = torch.Generator().manual_seed(seed)
    n = len(table['trial_type'])
    keep = {k: [] for k in ('outcome', 'loc', 'rt', 'pool0', 'pool1', 'early0', 'early1', 'dis0', 'dis1', 'snr0',
                            'snr1', 'snrzero0', 'snrzero1')}
    for s in range(0, n, 512):
        sub = {k: v[s:s + 512] for k, v in table.items()}
        inp = build_inputs(cfg, sub, 'cpu', generator=g)
        T = inp['t'].shape[0]
        out = net(inp, noise=True, generator=g)
        logits = net.hazard_logits(out['pool'])
        _, step, loc, _ = sample_saccades(logits, valid_mask(cfg, inp, T), generator=g)
        ev = evaluate(cfg, inp, step, loc, generator=g)
        keep['outcome'].append(ev['outcome'].numpy())
        keep['loc'].append(loc.numpy())
        keep['rt'].append(ev['rt'].numpy())
        t = inp['t'][None, :]
        on = inp['on'].min(1).values[:, None]
        delay = ((t >= on + 100) & (t < inp['t_go'][:, None])).float()
        early = ((t >= on + 50) & (t < on + 150)).float()
        pool = out['pool']                                   # [B,T,2]; pool l = location l
        for l in (0, 1):
            keep[f'pool{l}'].append(((pool[..., l] * delay).sum(1) / delay.sum(1).clamp(min=1)).numpy())
            keep[f'early{l}'].append(((pool[..., l] * early).sum(1) / early.sum(1).clamp(min=1)).numpy())
            nd = delay.sum(1).clamp(min=1)
            keep[f'dis{l}'].append(((out['dis'][..., l] * delay).sum(1) / nd).numpy())
            keep[f'snr{l}'].append(((out['snr'][..., l] * delay).sum(1) / nd).numpy())
            keep[f'snrzero{l}'].append((((out['snr'][..., l] <= 1e-6).float() * delay).sum(1) / nd).numpy())
    return {k: np.concatenate(v) for k, v in keep.items()}


def tables(cfg, seed):
    rec = from_recorded(str(BELIEF))
    rng = np.random.default_rng(500 + seed)
    syn = [make_session(cfg, rng, n_blocks=8) for _ in range(2)]
    cols = ['trial_type', 'id_L', 'id_R', 'sal_L', 'sal_R', 'rew_L', 'rew_R', 'V1', 'V2', 'fix_dur', 'delay',
            'onset_L', 'onset_R']
    tab = {c: np.concatenate([np.asarray(t[c], float) for t in [rec] + syn]) for c in cols}
    n = len(tab['V1'])
    tab['side_bias'] = np.zeros(n)
    tab['engaged'] = np.ones(n)
    return tab


def measures(tab, r):
    out = {}
    comp = r['outcome'] == 0
    fb = r['outcome'] == 1
    out['fixbreak'] = fb.mean()
    out['fixbreak_right'] = float(np.mean(r['loc'][fb] == 1)) if fb.sum() > 10 else np.nan
    out['completed'] = comp.mean()
    shown_loc = np.where(tab['id_R'] > 0, 1, 0)
    shown_id = np.where(tab['id_R'] > 0, tab['id_R'], tab['id_L'])
    v = np.where(shown_id == 1, tab['V1'], tab['V2']) / 100
    st = (tab['trial_type'] == 0)
    for l, name in ((1, 'inj'), (0, 'oth')):
        m = st & (shown_loc == l)
        ok = m & comp & ~np.isnan(r['rt'])
        out[f'st_completed_{name}'] = comp[m].mean()
        out[f'rt_{name}'] = np.median(r['rt'][ok]) if ok.sum() > 20 else np.nan
        if ok.sum() > 30:
            out[f'rho_{name}'] = spearmanr(v[ok], r['rt'][ok])[0]
            out[f'slope_{name}'] = np.polyfit(v[ok], r['rt'][ok], 1)[0]          # ms per 100 ms value
        else:
            out[f'rho_{name}'] = out[f'slope_{name}'] = np.nan
        p = r[f'pool{l}'][m]
        out[f'pool_{name}'] = p.mean()
        out[f'poolslope_{name}'] = np.polyfit(v[m], p, 1)[0] / max(p.mean(), 1e-6)  # fractional change / 100 ms
        # value carried by the upstream / SNr channel for this field, and SNr operating range (0 = floor)
        out[f'dis_rho_{name}'] = spearmanr(v[m], r[f'dis{l}'][m])[0] if np.std(r[f'dis{l}'][m]) > 0 else 0.0
        out[f'snr_{name}'] = r[f'snr{l}'][m].mean()
        out[f'snrfloor_{name}'] = r[f'snrzero{l}'][m].mean()
        # fixation breaks vs value (breaks after this field's target appears are not separable here: all breaks)
        fbm = fb[m].astype(float)
        out[f'fb_{name}'] = fbm.mean()
        out[f'fbslope_{name}'] = np.polyfit(v[m], fbm, 1)[0] if fbm.std() > 0 else 0.0
        e = r[f'early{l}'][m]
        out[f'earlyslope_{name}'] = np.polyfit(v[m], e, 1)[0] / max(e.mean(), 1e-6)
    ch = (tab['trial_type'] == 1) & comp
    y = (r['loc'][ch] == 1).astype(float)
    vr = np.where(tab['id_R'] == 1, tab['V1'], tab['V2'])[ch] / 100
    vl = np.where(tab['id_L'] == 1, tab['V1'], tab['V2'])[ch] / 100
    sal = np.where(tab['sal_R'][ch] > 0, 1.0, -1.0)
    out['p_right'] = y.mean() if len(y) else np.nan
    out['p_ch_completed'] = comp[tab['trial_type'] == 1].mean()
    try:
        f = sm.Logit(y, sm.add_constant(np.c_[vr, vl, sal])).fit(disp=0)
        out['b0'], out['b_vright'], out['b_vleft'], out['b_sal'] = f.params
    except Exception:
        out['b0'] = out['b_vright'] = out['b_vleft'] = out['b_sal'] = np.nan
    return out


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    OUT.mkdir(parents=True, exist_ok=True)
    specs = []
    if which in ('hand', 'all'):
        specs += [(f'hand_{a}', a, None) for a in HAND]
    if which in ('trained', 'all'):
        specs += [(n, n.split('_')[0], TRAINED / n / 'model.pt') for n in TRAINED_NETS]
    if which not in ('hand', 'trained', 'all'):           # comma-separated network names
        specs = [(n, n.split('_')[1], None) if n.startswith('hand_') else (n, n.split('_')[0], TRAINED / n / 'model.pt')
                 for n in which.split(',')]
        which = 'part_' + which.split(',')[0]
    rows = []
    for name, arch, path in specs:
        if path is None:
            net = build_handbuilt(arch, SRSConfig(device='cpu', shape_vis=SHAPE_VIS), **HAND[arch])
        else:
            net = load_net(str(path), device='cpu')
        for seed in range(n_seeds):
            tab = tables(net.cfg, seed)
            for cname, c in CONDS.items():
                if c is None:
                    reset_perturbation(net)
                else:
                    perturb(net, *c)
                r = run(net, tab, seed=1000 * seed + 7)
                reset_perturbation(net)
                rows.append({'net': name, 'arch': arch, 'seed': seed, 'cond': cname, 'n': len(tab['V1']),
                             **measures(tab, r)})
        print(name, 'done', flush=True)
        pd.DataFrame(rows).to_csv(OUT / f'runs_{which}.csv', index=False)


if __name__ == '__main__':
    main()
