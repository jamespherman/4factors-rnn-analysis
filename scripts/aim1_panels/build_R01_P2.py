#!/usr/bin/env python3
"""
Build R01 panel P2: alignment of factor effects across units (visual window, all units pooled, E and I).

Per-unit standardised effects (OLS coefficient / residual SD of sqrt spike count, 50-250 ms after target onset, target
at the unit's preferred location) are read unchanged from results/competition/p3_interactions/units.csv
(scripts/cp_p3_interactions.py). Statistic as in the original P3 report: Spearman r across all units (E + I pooled),
scipy p, units with both effects defined.
Pairs: image vs salience (stimulus-stimulus), expected (location prior) vs image (goal-stimulus), and the alternative
reward vs salience.
Run: python scripts/aim1_panels/build_R01_P2.py   (seconds)
"""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'results' / 'aim1_panels' / 'R01_core' / 'P2_factor_alignment_scatter'
P3 = REPO / 'results' / 'competition' / 'p3_interactions' / 'units.csv'
PAIRS = [('salience', 'image', 'a', 'stimulus-stimulus'), ('image', 'expected', 'b', 'goal-stimulus'),
         ('salience', 'reward', 'b_alt', 'goal-stimulus')]   # (x, y, panel, kind)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / 'scripts' / 'aim1_panels' / 'style.py', OUT / 'style.py')
    df = pd.read_csv(P3)
    cols = ['reward', 'expected', 'image', 'salience']
    u = df[['session', 'id', 'type', 'P'] + [f'visual_{c}' for c in cols]].rename(
        columns={'id': 'unit_id', 'type': 'unit_type', 'P': 'P_loc', **{f'visual_{c}': f'{c}_sd' for c in cols}})
    u.to_csv(OUT / 'P2_units.csv', index=False, float_format='%.6g')
    st = []
    for x, y, panel, kind in PAIRS:
        for cls in ('all', 'E', 'I'):
            g = u if cls == 'all' else u[u.unit_type == cls]
            d = g[[f'{x}_sd', f'{y}_sd']].dropna()
            r, p = spearmanr(d.iloc[:, 0], d.iloc[:, 1])
            st.append(dict(panel=panel, x_effect=x, y_effect=y, pair_kind=kind, unit_class=cls, spearman_r=r, p=p,
                           n=len(d), test='Spearman r across units (scipy p)' + (', pooled E+I (plotted)' if cls == 'all' else ', within class (not plotted)'),
                           window='visual 50-250 ms re target onset'))
    commit = subprocess.run(['git', '-C', str(REPO), 'rev-parse', '--short', 'HEAD'], capture_output=True,
                            text=True).stdout.strip()
    st = pd.DataFrame(st)
    st['session_filter'] = ('6 diagonal-target sessions (Newton 08_15 excluded), verified successful trials, unit-stable '
                            'trials, target at P, >= 40 trials; expected only for units that saw both block types')
    st['source'] = f'results/competition/p3_interactions/units.csv via scripts/aim1_panels/build_R01_P2.py, git {commit}'
    st.to_csv(OUT / 'stats.csv', index=False, float_format='%.5g')
    print(st[['panel', 'x_effect', 'y_effect', 'unit_class', 'spearman_r', 'p', 'n']].to_string())


if __name__ == '__main__':
    main()
