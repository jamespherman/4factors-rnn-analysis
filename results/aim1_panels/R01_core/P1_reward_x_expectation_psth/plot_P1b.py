"""P1b: simpler view of the reward x expectation interaction in classic SC neurons.
Expected-minus-unexpected target response (per unit, baseline-subtracted sp/s) for the high and the low reward cue,
mean +- SEM across units, aligned on target onset. Reads only P1_unit_psth.csv, P1_units.csv and style.py here."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style

HERE = Path(__file__).parent
style.apply()
u = pd.read_csv(HERE / 'P1_unit_psth.csv')
w = u.pivot_table(index=['session', 'unit_id', 'reward_cue', 'bin_center_ms'], columns='target',
                  values='rate_minus_baseline_sp_s').reset_index()
w['diff'] = w['expected'] - w['unexpected']
units = pd.read_csv(HERE / 'P1_units.csv')

fig, (ax, bx) = plt.subplots(1, 2, figsize=(3.6, 2.0), gridspec_kw=dict(width_ratios=[2.2, 1], wspace=0.55))
rows = []
for cue, col, lab in (('high', style.DATA, 'high-reward cue'), ('low', style.MUTED, 'low-reward cue')):
    m = w[w['reward_cue'] == cue].pivot_table(index=['session', 'unit_id'], columns='bin_center_ms', values='diff')
    t = m.columns.to_numpy(float)
    mu, se = m.mean().to_numpy(), m.sem().to_numpy()
    k = np.ones(3) / 3                                             # 3-bin (60 ms) boxcar for display only
    mus, ses = np.convolve(mu, k, 'same'), np.convolve(se, k, 'same')
    ax.fill_between(t, mus - ses, mus + ses, color=col, alpha=0.2, lw=0)
    ax.plot(t, mus, color=col, label=lab)
    rows += [dict(reward_cue=cue, bin_center_ms=a, diff_mean_sp_s=b, diff_sem_sp_s=c, smoothed_mean=d, smoothed_sem=e)
             for a, b, c, d, e in zip(t, mu, se, mus, ses)]
pd.DataFrame(rows).to_csv(HERE / 'P1b_expected_minus_unexpected.csv', index=False)
ax.axhline(0, color=style.MUTED, lw=0.6)
ax.axvline(0, color=style.MUTED, lw=0.6, ls=':')
ax.axvspan(50, 250, color=style.NULL, alpha=0.5, lw=0, zorder=0)
ax.set(xlim=(-300, 300), xlabel='time from target onset (ms)', ylabel='expected - unexpected (sp/s)')
ax.text(150, ax.get_ylim()[1], 'visual window', ha='center', va='top', fontsize=6, color=style.MUTED)
ax.legend(loc='lower left', fontsize=6)

d_hi = units['visual_50_250_high_expected_sp_s'] - units['visual_50_250_high_unexpected_sp_s']
d_lo = units['visual_50_250_low_expected_sp_s'] - units['visual_50_250_low_unexpected_sp_s']
for x, d, col in ((0, d_hi, style.DATA), (1, d_lo, style.MUTED)):
    bx.bar(x, d.mean(), yerr=d.sem(), color=col, width=0.6, error_kw=dict(lw=0.8))
bx.axhline(0, color=style.MUTED, lw=0.6)
bx.set_xticks([0, 1], ['high', 'low'])
bx.set(xlabel='reward cue', ylabel='expected - unexpected\n50-250 ms (sp/s)')
bx.set_title(f'n = {len(units)} classic units', fontsize=6.5)
pd.DataFrame(dict(reward_cue=['high', 'low'], mean_sp_s=[d_hi.mean(), d_lo.mean()],
                  sem_sp_s=[d_hi.sem(), d_lo.sem()], n_units=[len(units)] * 2)).to_csv(
    HERE / 'P1b_window_bars.csv', index=False)
style.save(fig, str(HERE / 'P1b_expected_minus_unexpected'))
