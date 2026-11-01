#!/usr/bin/env python3
"""Regenerate B5 from the CSVs in this folder (numpy/pandas/matplotlib only). Edit freely to restyle."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style

HERE = Path(__file__).resolve().parent
style.apply()
c = pd.read_csv(HERE / 'B5_coefs.csv')

# (model, target_cond, regressor, row label); None = group header
ROWS = [
    ('RF at the target', None),
    (('single', 'high', 'z_prior'), 'target at expected loc.'),
    (('single', 'low', 'z_target'), 'target at unexpected loc.'),
    ('RF at a non-target (competitor) site', None),
    (('single', 'low', 'z_prior'), 'expected loc., target elsewhere'),
    (('single', 'high', 'z_away'), 'opposite loc., target expected'),
    (('single', 'high', 'z_other'), 'non-prob. loc., target expected'),
    (('single', 'low', 'z_away_nontarget'), 'opposite loc., target elsewhere'),
    (('single', 'low', 'z_other_nontarget'), 'non-prob. loc., target elsewhere'),
    ('Joint model (partial coefficients)', None),
    (('joint (z_prior + z_other)', 'low', 'z_prior'), 'expected loc. | non-prob., target elsewhere'),
    (('joint (z_prior + z_other)', 'low', 'z_other'), 'non-prob. loc. | expected, target there 1/3'),
    (('joint (z_prior + z_other)', 'high', 'z_other'), 'non-prob. loc. | expected, target expected'),
    (('joint (z_prior + z_other)', 'high', 'z_prior'), 'expected loc. (= target) | non-prob.'),
    ('All units', None),
    (('single', 'high', 'z_all'), 'target at expected loc.'),
    (('single', 'low', 'z_all'), 'target elsewhere'),
]


def pfmt(p):
    return 'p<0.001' if p < 0.001 else f'p={p:.2g}'


fig, ax = plt.subplots(figsize=(3.6, 3.55))
y, ticks, labels = 0, [], []
for key, lab in ROWS:
    if lab is None:
        y += 0.4
        ax.text(-14.6, y, key, fontsize=6.5, fontweight='bold', va='center', ha='left')
        y += 1
        continue
    r = c[(c.model == key[0]) & (c.target_cond == key[1]) & (c.regressor == key[2])].iloc[0]
    comp = r.competition_role == 'competitor'
    if comp:
        ax.fill_between([0, 6], y - 0.5, y + 0.5, color=style.RESOLVER, alpha=0.13, lw=0, zorder=0)
    ax.errorbar([r.coef_pct_per_sd], [y], xerr=[[r.coef_pct_per_sd - r.ci95_lo], [r.ci95_hi - r.coef_pct_per_sd]],
                fmt='o', color=style.DATA, mfc=style.DATA if r.p_ols < 0.05 else 'white', ms=3.5, elinewidth=1.0,
                capsize=0, zorder=3)
    ax.text(6.2, y, f'{r.coef_pct_per_sd:+.1f} ({pfmt(r.p_ols)}, n={int(r.n_trials)})', fontsize=5.5, va='center',
            color=style.DATA)
    ticks.append(y)
    labels.append(lab)
    y += 1
ax.axvline(0, color=style.MUTED, lw=0.6, zorder=1)
ax.set_yticks(ticks)
ax.set_yticklabels(labels, fontsize=6)
ax.set_ylim(y - 0.3, -0.2)
ax.set_xlim(-11.5, 6)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='y', length=0)
ax.set_xlabel('RT change (% per SD of pre-target rate)')
ax.text(3.0, -0.1, 'local competition\npredicts slower RT', fontsize=5.5, color=style.RESOLVER, ha='center', va='bottom')
ax.text(-5.5, -0.1, 'faster', fontsize=5.5, color=style.MUTED, ha='center', va='bottom')
fig.text(0.0, -0.01, 'Mean ± 95% CI (OLS log RT ~ session + target loc. + reward + z; x100). Filled: p<0.05. '
         'Shading: half-plane a\nlocal competition between SC sites predicts for non-target (competitor) groups. '
         'Joint model of the source script\n(z_prior + z_away) is not estimable: the two groups never co-occur on a trial.',
         fontsize=5.2, color=style.MUTED, va='top')
style.save(fig, str(HERE / 'B5_rt_state_specificity'))
