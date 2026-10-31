#!/usr/bin/env python3
"""Summarise results/hub_models/bicuculline/runs_*.csv (scripts/srs_bicuculline.py): per architecture and condition,
median [range] over networks of the per-network seed means; occlusion indices relative to each network's control."""

from pathlib import Path

import numpy as np
import pandas as pd

D = Path(__file__).parent.parent / 'results' / 'hub_models' / 'bicuculline'
COLS = ['fixbreak', 'fixbreak_right', 'p_right', 'b0', 'b_vright', 'b_vleft', 'rt_inj', 'rt_oth', 'rho_inj', 'rho_oth',
        'slope_inj', 'slope_oth', 'poolslope_inj', 'poolslope_oth', 'dis_rho_inj', 'snrfloor_inj']


def main():
    df = pd.concat([pd.read_csv(f) for f in D.glob('runs_*.csv')], ignore_index=True)
    df['group'] = np.where(df['net'].str.startswith('hand_'), 'hand_' + df['arch'], df['arch'])
    df['abspool_inj'] = df['poolslope_inj'] * df['pool_inj']          # absolute pool-rate change per 100 ms value
    df['abspool_oth'] = df['poolslope_oth'] * df['pool_oth']
    m = df.groupby(['group', 'net', 'cond'])[COLS + ['abspool_inj', 'abspool_oth']].mean().reset_index()
    ctrl = m[m['cond'] == 'ctrl'].set_index('net')
    for c in ('abspool_inj', 'abspool_oth', 'slope_inj', 'poolslope_inj', 'slope_oth', 'poolslope_oth', 'b_vright', 'b_vleft'):
        m[c + '_idx'] = m[c] / m['net'].map(ctrl[c])
    m['drt_inj'] = m['rt_inj'] - m['net'].map(ctrl['rt_inj'])
    m['drt_oth'] = m['rt_oth'] - m['net'].map(ctrl['rt_oth'])
    m['db0'] = m['b0'] - m['net'].map(ctrl['b0'])
    m['dp_right'] = m['p_right'] - m['net'].map(ctrl['p_right'])
    m.to_csv(D / 'per_network.csv', index=False)
    show = ['fixbreak', 'fixbreak_right', 'drt_inj', 'drt_oth', 'dp_right', 'db0', 'b_vright_idx', 'b_vleft_idx', 'rho_inj',
            'slope_inj_idx', 'poolslope_inj_idx', 'abspool_inj_idx', 'poolslope_oth_idx', 'dis_rho_inj', 'snrfloor_inj']
    order = ['ctrl', 'full05', 'full10', 'full20', 'full35', 'local20', 'snr20', 'snr50', 'clamp', 'musc50']
    L = ['# In-silico SC bicuculline (left SC = right field): median [min, max] over networks\n',
         'idx = value-dependence under the manipulation / control for the same network (1 = unchanged, 0 = abolished).',
         'drt = RT change (ms), db0 = choice-bias change toward the injected field (logit).\n']
    for g, gg in m.groupby('group'):
        L.append(f'## {g} ({gg["net"].nunique()} networks)')
        L.append('| cond | ' + ' | '.join(show) + ' |')
        L.append('|' + '---|' * (len(show) + 1))
        for c in order:
            x = gg[gg['cond'] == c]
            if not len(x):
                continue
            cells = []
            for s in show:
                v = x[s].dropna()
                cells.append('' if not len(v) else (f'{v.median():+.2f}' if len(v) == 1 else
                                                    f'{v.median():+.2f} [{v.min():+.2f},{v.max():+.2f}]'))
            L.append(f'| {c} | ' + ' | '.join(cells) + ' |')
        L.append('')
    (D / 'summary.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
