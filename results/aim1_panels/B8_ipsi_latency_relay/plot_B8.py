#!/usr/bin/env python3
"""B8: is there a commissural relay delay in ipsilateral visual responses?
Reads only the CSVs in this folder (plus style.py). Run: python plot_B8.py"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import style

HERE = Path(__file__).resolve().parent
style.apply()

P = pd.read_csv(HERE / 'B8_pairs.csv')
cal = pd.read_csv(HERE / 'B8_calibration.csv')
calfit = pd.read_csv(HERE / 'B8_calibration_fit.csv')
bands = pd.read_csv(HERE / 'B8_bands.csv').set_index('band')
stats = pd.read_csv(HERE / 'stats.csv').set_index('statistic')
relay = bands.loc['commissural_relay_prediction']
ci = bands.loc['excess_median_CI_unit_bootstrap']
med = stats.loc['relay excess, median (ms) [unit bootstrap CI]', 'value']
n_units = int(stats.loc['paired conditions: units', 'value'])
slope = stats.loc['calibration slope (latency per ln amplitude)', 'value']
CLASSES = [('I', style.DATA, 'non-classic (I)'), ('E', 'none', 'classic (E)')]


def relay_band(ax):
    ax.axhspan(relay.low_ms, relay.high_ms, color=style.RESOLVER, alpha=0.18, lw=0)
    for y in (relay.low_ms, relay.high_ms):
        ax.axhline(y, color=style.RESOLVER, ls='--', lw=0.8)
    ax.axhline(0, color=style.MUTED, lw=0.6)


fig = plt.figure(figsize=(3 * style.PANEL[0] + 0.5, style.PANEL[1] + 0.1))
gs = fig.add_gridspec(1, 4, width_ratios=[0.62, 0.55, 1.0, 1.0], wspace=0.75)
axA = fig.add_subplot(gs[0])
axZ = fig.add_subplot(gs[1])
axB = fig.add_subplot(gs[2])
axC = fig.add_subplot(gs[3])

# a (left): every condition, full range
relay_band(axA)
rng = np.random.default_rng(0)
for t, mfc, lab in CLASSES:
    g = P[P.unit_type == t]
    axA.scatter(rng.uniform(-0.3, 0.3, len(g)), g.excess_ms.to_numpy(), s=6, facecolors=mfc,
                edgecolors=style.DATA, lw=0.5, alpha=0.6, label=f'{lab}, n = {len(g)}')
axA.set_xlim(-0.5, 0.5)
axA.set_xticks([])
axA.set_ylabel('Ipsi minus contra onset latency,\namplitude-corrected (ms)')
axA.set_title(f'a  Relay excess\n    {len(P)} conditions, {n_units} units', loc='left')
axA.legend(loc='upper left', bbox_to_anchor=(-0.1, 1.0), fontsize=5.2, handletextpad=0.1, borderaxespad=0.0)
axA.set_ylim(-140, 200)

# a (right): zoom on the median vs the relay band
relay_band(axZ)
axZ.errorbar([0], [med], yerr=[[med - ci.low_ms], [ci.high_ms - med]], fmt='o', color=style.DATA, ms=4,
             capsize=2.5, lw=1.4)
axZ.text(0.12, med, f'median\n{med:+.1f} ms\n[{ci.low_ms:+.1f}, {ci.high_ms:+.1f}]', fontsize=5.6, va='center')
axZ.text(-0.45, relay.high_ms + 1.5, 'commissural relay\nprediction', color=style.RESOLVER, fontsize=5.8,
         va='bottom')
axZ.set_xlim(-0.5, 1.0)
axZ.set_xticks([])
axZ.set_ylim(-10, 36)
axZ.set_ylabel('Median excess (ms)', labelpad=1)
axZ.set_title('zoom', loc='left', fontsize=6.5, color=style.MUTED)

# b: raw latencies (secondary; not amplitude-corrected)
lim = (20, 250)
xx = np.array(lim)
axB.fill_between(xx, xx + relay.low_ms, xx + relay.high_ms, color=style.RESOLVER, alpha=0.18, lw=0)
axB.plot(xx, xx, color=style.MUTED, lw=0.8)
for t, mfc, lab in CLASSES:
    g = P[P.unit_type == t]
    axB.scatter(g.lat_contra_ms.to_numpy(), g.lat_ipsi_ms.to_numpy(), s=6, facecolors=mfc, edgecolors=style.DATA,
                lw=0.5, alpha=0.6)
axB.set_xlim(*lim)
axB.set_ylim(*lim)
axB.set_aspect('equal')
axB.set_xlabel('Contra onset latency (ms)')
axB.set_ylabel('Ipsi 90°-neighbour onset latency (ms)')
axB.set_title('b  Raw latencies (not\n    amplitude-corrected)', loc='left')

# c: latency-amplitude calibration
axC.scatter(cal.amp_sp_s.to_numpy(), cal.lat_ms.to_numpy(), s=5, facecolors='none', edgecolors=style.MUTED,
            lw=0.5, alpha=0.7)
axC.plot(calfit.amp_sp_s.to_numpy(), calfit.fit_lat_ms.to_numpy(), color=style.DATA, lw=1.2)
axC.set_xscale('log')
axC.set_xlabel('Response amplitude (sp/s)')
axC.set_ylabel('Onset latency (ms)')
axC.set_title('c  Calibration\n    (classic units, contra)', loc='left')
axC.text(0.97, 0.97, f'{slope:.1f} ms per ln unit\nr = {stats.loc["calibration r", "value"]:.2f}, '
                     f'n = {len(cal)}', transform=axC.transAxes, ha='right', va='top', fontsize=5.8)

style.save(fig, str(HERE / 'B8_ipsi_latency_relay'))
print('saved', HERE / 'B8_ipsi_latency_relay.png')
