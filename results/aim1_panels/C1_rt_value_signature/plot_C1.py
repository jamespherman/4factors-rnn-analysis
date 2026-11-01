#!/usr/bin/env python3
"""C1: single-target RT vs value (S9) in task-trained / hand-built srsClassic networks and the monkey (08-20, 08-19).
Reads only the CSVs in this folder (plus style.py). Run: python plot_C1.py  ->  C1_rt_value_signature.png / .pdf"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import style

HERE = Path(__file__).resolve().parent
style.apply()
net = pd.read_csv(HERE / 'C1_ab_s9_per_network.csv')
rec = pd.read_csv(HERE / 'C1_ab_recorded_s9.csv')
rob = pd.read_csv(HERE / 'C1_c_rt_shift_robustness.csv')
dv = pd.read_csv(HERE / 'C1_d_rt_vs_true_value.csv')
ev = pd.read_csv(HERE / 'C1_e_rt_vs_belief_0820_sequence.csv')
ARCHS = ['A', 'AE', 'B', 'C']
r20 = rec[(rec.session == '08-20') & (rec.value == 'belief')].iloc[0]
b20 = rec[(rec.session == '08-20') & (rec.value == 'block_mean')].iloc[0]
r19 = rec[(rec.session == '08-19') & (rec.value == 'belief')].iloc[0]
YL = (-0.92, 0.2)


def recorded_band(ax):
    """08-20 S9 vs fitted belief (line) and its 95% bootstrap CI (band)."""
    ax.axhspan(r20.boot_lo, r20.boot_hi, color=style.NULL, lw=0, zorder=0)
    ax.axhline(r20.rho, color=style.DATA, lw=1.0, zorder=1)
    ax.axhline(0, color=style.MUTED, lw=0.5, zorder=0)


def monkey_column(ax, x0=4.3):
    """Recorded S9 with 95% bootstrap CI: 08-20 vs fitted belief (filled), 08-20 vs true block-mean value (open,
    drift control), 08-19 vs belief (grey)."""
    pts = ((-0.25, r20, style.DATA, style.DATA), (0.0, b20, 'white', style.DATA), (0.25, r19, style.MUTED, style.MUTED))
    for dx, r, fc, ec in pts:
        ax.errorbar([x0 + dx], [r.rho], yerr=[[r.rho - r.boot_lo], [r.boot_hi - r.rho]], fmt='o', ms=3.5, mfc=fc,
                    mec=ec, ecolor=ec, elinewidth=0.8, capsize=0, zorder=4)


def dots(ax, col, title):
    recorded_band(ax)
    rng = np.random.default_rng(1)
    for i, a in enumerate(ARCHS):
        g = net[(net.arch == a) & ~net.handbuilt]
        x = i - 0.08 + rng.uniform(-0.11, 0.11, len(g))
        ax.scatter(x, g[col].to_numpy(), s=11, color=style.ARCH[a], edgecolor='white', lw=0.3, zorder=3)
        ax.plot([i - 0.22, i + 0.06], [g[col].median()] * 2, color=style.ARCH[a], lw=1.2, zorder=2)
        h = net[(net.arch == a) & net.handbuilt]
        if len(h):
            ax.scatter([i + 0.24] * len(h), h[col].to_numpy(), s=16, marker='D', facecolor='white',
                       edgecolor=style.ARCH[a], lw=0.9, zorder=3)
    monkey_column(ax)
    ax.axvline(3.65, color=style.MUTED, lw=0.4, zorder=0)
    ax.set_xticks(list(range(4)) + [4.3], ARCHS + ['monkey'])
    ax.set_xlim(-0.5, 4.75)
    ax.set_ylim(*YL)
    ax.set_ylabel('S9: Spearman ρ(RT, value)')
    ax.set_title(title, loc='left')


fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.6), gridspec_kw=dict(hspace=0.55, wspace=0.5))
a1, a2, a3 = axs[0]
a4, a5, a6 = axs[1]

dots(a1, 's9_rho', 'a  Models on task value range')
dots(a2, 's9_rho_recseq', "b  Models on monkey's 08-20 beliefs")

# c: robustness to slower RTs (B, C; raised readout threshold)
recorded_band(a3)
for n, g in rob.groupby('net'):
    g = g.sort_values('theta_offset')
    a3.plot(g.delta_rt_ms_from_offset0.to_numpy(), g.s9_rho.to_numpy(), color=style.ARCH[g.arch.iloc[0]], lw=0.9,
            marker='o', ms=2)
a3.set_ylim(-0.55, 0.1)
a3.set_xlabel('RT slowing, raised threshold (ms)')
a3.set_ylabel('S9')
a3.set_title('c  Robust to slower RTs (B, C)', loc='left')
a3.text(0.98, 0.03, 'B (6 nets)', color=style.PRIORITY, transform=a3.transAxes, ha='right', fontsize=6)
a3.text(0.98, 0.94, 'C (3 nets)', color=style.OUTCOME, transform=a3.transAxes, ha='right', fontsize=6)


def curves(ax, d, title, xlabel):
    for a in ('A', 'B', 'C'):
        g = d[d.arch == a].sort_values('value_mean_ms')
        x, y = g.value_mean_ms.to_numpy(), g.rt_pct_change_from_mean.to_numpy()
        ax.fill_between(x, g.ci95_lo.to_numpy(), g.ci95_hi.to_numpy(), color=style.ARCH[a], alpha=0.18, lw=0)
        ax.plot(x, y, color=style.ARCH[a], lw=1.3, marker='o', ms=2.5)
    m = d[d.arch == 'data'].sort_values('value_mean_ms')
    mx, my = m.value_mean_ms.to_numpy(), m.rt_pct_change_from_mean.to_numpy()
    ax.errorbar(mx, my, yerr=[my - m.ci95_lo.to_numpy(), m.ci95_hi.to_numpy() - my], color=style.DATA, lw=1.3,
                elinewidth=0.7, marker='o', ms=3, capsize=0, zorder=4)
    w = d[d.arch == 'data_within_block'].sort_values('value_mean_ms')
    if len(w):
        ax.plot(w.value_mean_ms.to_numpy(), w.rt_pct_change_from_mean.to_numpy(), color=style.DATA, lw=1.0, ls='--',
                marker='o', ms=2.5, mfc='white', zorder=4)
    ax.axhline(0, color=style.MUTED, lw=0.5, zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('RT, % change from mean')
    ax.set_title(title, loc='left')


curves(a4, dv, 'd  RT vs true block value', 'value of shown target (ms juice)')
curves(a5, ev, 'e  RT vs fitted belief, 08-20', 'belief value of shown target (ms)')
a4.set_ylim(-8, 8)
a5.set_ylim(-12, 9)

a6.axis('off')
h = [Line2D([], [], color=style.RESOLVER, marker='o', lw=0, ms=4, label='A / AE resolver'),
     Line2D([], [], color=style.PRIORITY, marker='o', lw=0, ms=4, label='B priority hub'),
     Line2D([], [], color=style.OUTCOME, marker='o', lw=0, ms=4, label='C outcome hub'),
     Line2D([], [], color=style.MUTED, marker='D', mfc='white', lw=0, ms=4, label='hand-built network'),
     Line2D([], [], color=style.DATA, marker='o', lw=1.0, ms=3.5,
            label=f'08-20 vs fitted belief: ρ = {r20.rho:.2f} (n = {r20.n:.0f})'),
     plt.Rectangle((0, 0), 1, 1, color=style.NULL, label='  its 95% bootstrap CI'),
     Line2D([], [], color=style.DATA, marker='o', mfc='white', lw=0, ms=3.5,
            label=f'08-20 vs true block value: ρ = {b20.rho:+.2f}'),
     Line2D([], [], color=style.DATA, marker='o', mfc='white', lw=1.0, ls='--', ms=3,
            label='08-20, block-mean RT removed (e)'),
     Line2D([], [], color=style.MUTED, marker='o', lw=0, ms=3.5,
            label=f'08-19 vs belief: ρ = {r19.rho:+.2f} (n = {r19.n:.0f})')]
a6.legend(handles=h, loc='upper left', fontsize=5.8, handlelength=1.6, borderaxespad=0, labelspacing=0.45)
a6.text(0, 0.24, 'Models w = 1 (value on every trial). d, e: networks\nclosest to their architecture median '
        '(A_s2, B_s0,\nC_s5). RT normalised to each source\'s mean\n(models ~110-150 ms, monkey 221 ms).',
        fontsize=5.5, color=style.MUTED, transform=a6.transAxes, va='top')
style.save(fig, str(HERE / 'C1_rt_value_signature'))
