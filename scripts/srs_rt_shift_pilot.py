#!/usr/bin/env python3
"""
Pilot for plans/2026-10-31_retrain_rt_plan.md (review point 4): do trained v2 priority-hub (B) and outcome-hub (C)
networks show the S9 RT-value contrast once their RTs are moved off the floor post hoc (raising the hazard threshold
theta), with value engaged on every trial (w = 1)? Also reports how graded each network's upstream signal is on
single-target trials (Spearman of delay-window disinhibition at the target's location with its value).
Writes results/hub_models/rt_shift_pilot/summary.md.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
import srs_bicuculline as sb  # noqa: E402

OUT = ROOT / 'results' / 'hub_models' / 'rt_shift_pilot'
NETS = [f'B_s{s}' for s in range(6)] + ['C_s0', 'C_s4', 'C_s5']
OFFS = (0.0, 0.1, 0.2, 0.35, 0.5)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in NETS:
        net = sb.load_net(str(sb.TRAINED / name / 'model.pt'))
        th0 = float(net.theta.data)
        for off in OFFS:
            net.theta.data.fill_(th0 + off)
            for seed in range(2):
                tab = sb.tables(net.cfg, seed)
                r = sb.run(net, tab, seed=1000 * seed + 7)
                st = tab['trial_type'] == 0
                loc = np.where(tab['id_R'] > 0, 1, 0)
                sid = np.where(tab['id_R'] > 0, tab['id_R'], tab['id_L'])
                v = np.where(sid == 1, tab['V1'], tab['V2'])
                ok = st & (r['outcome'] == 0) & np.isfinite(r['rt'])
                dis = np.where(loc == 1, r['dis1'], r['dis0'])
                rows.append(dict(net=name, arch=name[0], off=off, seed=seed, rt=np.median(r['rt'][ok]),
                                 rt_iqr=np.subtract(*np.percentile(r['rt'][ok], [75, 25])),
                                 completed=(r['outcome'][st] == 0).mean(), fixbreak=(r['outcome'] == 1).mean(),
                                 s9=spearmanr(v[ok], r['rt'][ok])[0], dis_rho=spearmanr(v[st], dis[st])[0]))
        net.theta.data.fill_(th0)
        print(name, 'done', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'runs.csv', index=False)
    m = df.groupby(['arch', 'net', 'off']).mean(numeric_only=True).reset_index()
    L = ['# RT-shift pilot: S9 (Spearman RT vs shown value, single-target, w = 1) after raising the hazard threshold\n',
         'Recorded 08-20: median RT 209 ms, S9 -0.21.\n']
    for a, g in m.groupby('arch'):
        L.append(f'## {a}')
        L.append('| theta offset | median RT (ms), range over nets | S9 median [min, max] | upstream dis-value rho | completed |')
        L.append('|---|---|---|---|---|')
        for off, x in g.groupby('off'):
            L.append(f'| +{off:.2f} | {x.rt.median():.0f} [{x.rt.min():.0f}, {x.rt.max():.0f}] | '
                     f'{x.s9.median():+.2f} [{x.s9.min():+.2f}, {x.s9.max():+.2f}] | {x.dis_rho.median():+.2f} | '
                     f'{x.completed.median():.2f} |')
        L.append('')
    (OUT / 'summary.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
