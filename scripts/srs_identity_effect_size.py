#!/usr/bin/env python3
"""Per-unit identity effect beyond value (block-level contrast, as bilateral-SC-recordings computes it on the recordings):
for each block, log(mean rate T1 / mean rate T2) on single-target trials with the target contralateral to the unit;
identity effect = mean over blocks (value cancels across the alternating rich identity). Noise-free model rates mapped
to recorded-like sp/s (base ~3, modulation ~15). Reports |delta| per unit for hand-built and trained A, B, C."""
import sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'external' / 'conflict_srs'))
from srs import SRSConfig, build_handbuilt, simulate
from srs.task import from_recorded
sys.path.insert(0, str(ROOT)); from src.srs_load import load_net
torch.set_num_threads(1)
tab = from_recorded(str(ROOT / 'results/hub_models/belief/08-20_trials_with_belief.csv'))
blocks = np.cumsum(np.r_[0, np.diff(tab['block_type']) != 0])
nets = {'A_hand': build_handbuilt('A', SRSConfig(device='cpu')), 'B_hand': build_handbuilt('B', SRSConfig(device='cpu')),
        'C_hand': build_handbuilt('C', SRSConfig(device='cpu'), c_gain=1.1)}
for a in 'ABC':
    nets[f'{a}_s0'] = load_net(str(ROOT / f'results/hub_models/trained/{a}_s0/model.pt'))
rng = np.random.default_rng(0)
rows = []
import pandas as pd
print('label  window  median|delta|  frac>0.1  frac>0.2  (units contralateral to target; E_pool | all)')
for lab, net in nets.items():
    res = simulate(net, tab, seed=0)
    t, sc, units = res['table'], res['rates']['sc'].astype(float), res['units']
    side = np.asarray(units['sc_side']); typ = np.asarray(units['type'])
    tavg = np.nanmean(sc, 0)
    base = np.exp(np.log(3) + 0.6 * rng.standard_normal(sc.shape[2]))
    gain = np.exp(np.log(15) + 0.7 * rng.standard_normal(sc.shape[2])) / np.maximum(np.nanmax(tavg, 0) - np.nanmin(tavg, 0), 1e-3)
    rate = base + gain * (sc - np.nanmin(tavg, 0))
    tgt = np.where(np.isnan(t['onset_step_L']), t['onset_step_R'], np.fmin(t['onset_step_L'], t['onset_step_R'])).astype(int)
    st = (t['outcome'] == 0) & (t['trial_type'] == 0)
    shown_loc = np.where(t['id_R'] > 0, 1, 0); shown_id = np.where(t['id_R'] > 0, t['id_R'], t['id_L'])
    for wname, a, b in (('delay', 10, 45), ('visual', 4, 10), ('presac', None, None)):
        if a is None:                                                          # -100..0 ms re saccade onset
            sac = (t['go_step'] + np.round(np.nan_to_num(t['rt'], nan=0) / 10)).astype(int)
            win = np.array([np.nanmean(rate[k, max(sac[k] - 10, 0):max(sac[k], 1)], 0) for k in range(len(tgt))])
        else:
            win = np.array([np.nanmean(rate[k, tgt[k] + a:tgt[k] + b], 0) for k in range(len(tgt))])
        for grp, mask in (('E_pool', typ == 'E_pool'), ('all', np.ones(len(typ), bool))):
            deltas = []
            for s_name, loc in (('L', 1), ('R', 0)):
                for u in np.flatnonzero((side == s_name) & mask):
                    d = []
                    for bk in np.unique(blocks):
                        m = st & (shown_loc == loc) & (blocks == bk)
                        r1, r2 = win[m & (shown_id == 1), u], win[m & (shown_id == 2), u]
                        if len(r1) > 2 and len(r2) > 2:
                            d.append(np.log(np.mean(r1) / np.mean(r2)))
                    if d:
                        deltas.append(np.mean(d))
            deltas = np.array(deltas)
            rows.extend({'model': lab, 'window': wname, 'group': grp, 'delta_true': x} for x in deltas)
            deltas = np.abs(deltas)
            print(f'{lab:7s} {wname:6s} {grp:6s} {np.median(deltas):.3f}  {np.mean(deltas > 0.1):.2f}  {np.mean(deltas > 0.2):.2f}')

out = ROOT / 'results' / 'hub_models' / 'identity_delta_true.csv'
pd.DataFrame(rows).to_csv(out, index=False)
print('saved', out)
