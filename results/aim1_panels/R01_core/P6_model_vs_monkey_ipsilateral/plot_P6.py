"""P6: ipsilateral activation, task-trained local SC model vs monkey SC. Reads only P6_monkey.csv, P6_model.csv and
style.py in this folder. Run: python plot_P6.py"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style

HERE = Path(__file__).parent
style.apply()
mk = pd.read_csv(HERE / 'P6_monkey.csv')
md = pd.read_csv(HERE / 'P6_model.csv')
mk = mk[mk['monkey'] == 'both']

fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.2), sharey=True, gridspec_kw=dict(wspace=0.15))
spec = [('putative interneurons\nvs model I units', 'putative interneuron', ['I commissural', 'I local']),
        ('classic neurons\nvs model E units', 'classic', ['E'])]
w = 0.26
for ax, (title, dcls, mcls) in zip(axes, spec):
    for xi, side in enumerate(('contra', 'ipsi')):
        d = mk[(mk['unit_class'] == dcls) & (mk['side'] == side)].iloc[0]
        ax.bar(xi - w, 100 * d['frac'], w, color=style.DATA,
               yerr=[[100 * (d['frac'] - d['ci95_lo'])], [100 * (d['ci95_hi'] - d['frac'])]],
               error_kw=dict(lw=0.8, ecolor=style.MUTED), label='monkey SC' if xi == 0 else None)
        for off, grp, hatch, lab in ((0, 'trained (bilateral)', None, 'task-trained local SC model'),
                                     (w, 'trained + cross-midline E->I', '////', '+ cross-midline excitation')):
            s = md[(md['group'] == grp) & (md['unit_class'].isin(mcls)) & (md['side'] == side)]
            m, sd = s['frac_mean'].mean(), s['frac_sd'].mean()
            ax.bar(xi + off, 100 * m, w, color='white' if hatch else style.RESOLVER, edgecolor=style.RESOLVER,
                   hatch=hatch, lw=0.8, yerr=100 * sd, error_kw=dict(lw=0.8, ecolor=style.MUTED),
                   label=lab if xi == 0 else None)
    ax.set_xticks([0, 1], ['contralateral\ntarget', 'ipsilateral\ntarget'])
    ax.set_title(title, fontsize=6.8)
    ax.set_ylim(0, 105)
axes[0].set_ylabel('units activated by target (%)')
axes[0].legend(loc='upper right', fontsize=5.8, handlelength=1.2)
style.save(fig, str(HERE / 'P6_model_vs_monkey_ipsilateral'))
