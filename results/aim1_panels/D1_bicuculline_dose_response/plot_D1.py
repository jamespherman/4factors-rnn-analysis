#!/usr/bin/env python3
"""D1: in-silico SC bicuculline / muscimol dose-response per architecture. Reads only the CSVs in this folder (plus
style.py). Run: python plot_D1.py  ->  D1_bicuculline_dose_response.png / .pdf"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import style

HERE = Path(__file__).resolve().parent
style.apply()
L = pd.read_csv(HERE / 'D1_per_network_long.csv')
S = pd.read_csv(HERE / 'D1_architecture_summary.csv')
ARCHS = ['A', 'AE', 'B', 'C']
LS = {'A': '-', 'AE': (0, (3, 1.5)), 'B': '-', 'C': '-'}   # A and AE share the resolver colour; AE dashed
NAME = {'A': 'A resolver', 'AE': 'AE resolver', 'B': 'B priority hub', 'C': 'C outcome hub'}


def arch_lines(ax, measure, scale=1.0, shade=True, ls_override=None):
    for a in ARCHS:
        g = S[(S.measure == measure) & (S.arch == a) & (S.manipulation != 'muscimol') & S.plotted].sort_values('dose_d')
        x = g.dose_d.to_numpy()
        if shade:
            ax.fill_between(x, scale * g['min'].to_numpy(), scale * g['max'].to_numpy(), color=style.ARCH[a],
                            alpha=0.13, lw=0)
        ax.plot(x, scale * g['median'].to_numpy(), color=style.ARCH[a], lw=1.4,
                ls=ls_override if ls_override else LS[a])


def hand(ax, measure, scale=1.0, ls='-'):
    for a in ('A', 'B', 'C'):
        g = L[(L.measure == measure) & (L.net == f'hand_{a}') & (L.manipulation != 'muscimol')].dropna(
            subset=['value']).sort_values('dose_d')
        ax.plot(g.dose_d.to_numpy(), scale * g.value.to_numpy(), color=style.ARCH[a], lw=0, marker='D', ms=2.8,
                mfc='white', mew=0.8, zorder=4)


fig = plt.figure(figsize=(7.2, 4.4))
gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.55], hspace=0.6, wspace=0.6)
a1 = fig.add_subplot(gs[0, 0]); a2 = fig.add_subplot(gs[0, 1]); a3 = fig.add_subplot(gs[0, 2])
a4 = fig.add_subplot(gs[1, 0]); a5 = fig.add_subplot(gs[1, 3]); a6 = fig.add_subplot(gs[0, 3])
leg = fig.add_subplot(gs[1, 1:3]); leg.axis('off')
DOSES = [0, 0.05, 0.1, 0.2, 0.35]

# a: RT change into injected field; b: other field
arch_lines(a1, 'rt_change_injected_field_ms')
hand(a1, 'rt_change_injected_field_ms')
a1.set_title('a  RT, injected field', loc='left')
a1.set_ylabel('RT change (ms)')
a2.set_title('b  RT, other field', loc='left')
arch_lines(a2, 'rt_change_other_field_ms')
hand(a2, 'rt_change_other_field_ms')
a2.set_ylim(a1.get_ylim())
a2.set_ylabel('RT change (ms)')

# c: fixation breaks (%)
arch_lines(a3, 'fixation_break_rate', scale=100)
hand(a3, 'fixation_break_rate', scale=100)
a3.set_title('c  Fixation breaks', loc='left')
a3.set_ylabel('% of trials')

# d: choice bias, Δ P(choose injected field)
arch_lines(a4, 'delta_p_choose_injected_field', scale=100)
hand(a4, 'delta_p_choose_injected_field', scale=100)
a4.set_title('d  Choice toward injected field', loc='left')
a4.set_ylabel('Δ P(choose injected), % pts')

for ax in (a1, a2, a3, a4):
    ax.axhline(0, color=style.MUTED, lw=0.5, zorder=0)
    ax.set_xticks(DOSES, ['0', '.05', '.1', '.2', '.35'])
    ax.set_xlabel('bicuculline dose d')
    ax.set_xlim(-0.015, 0.365)


def musc(ax, measure, scale, ylabel, title):
    rng = np.random.default_rng(3)
    for i, a in enumerate(ARCHS):
        g = L[(L.measure == measure) & (L.arch == a) & (L.manipulation == 'muscimol') & ~L.handbuilt].dropna(
            subset=['value'])
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(g)), scale * g.value.to_numpy(), s=9, color=style.ARCH[a],
                   edgecolor='white', lw=0.3, zorder=3)
        h = L[(L.measure == measure) & (L.net == f'hand_{a}') & (L.manipulation == 'muscimol')].dropna(subset=['value'])
        if len(h):
            ax.scatter([i + 0.3] * len(h), scale * h.value.to_numpy(), s=12, marker='D', facecolor='white',
                       edgecolor=style.ARCH[a], lw=0.8, zorder=3)
    ax.axhline(0, color=style.MUTED, lw=0.5, zorder=0)
    ax.set_xticks(range(4), ARCHS)
    ax.set_xlim(-0.5, 3.6)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc='left')


musc(a6, 'rt_change_injected_field_ms', 1, 'RT change (ms)', 'e  Muscimol d 0.5')
a6.set_ylim(0, 260)
musc(a5, 'delta_p_choose_injected_field', 100, 'Δ P(choose inj.), % pts', 'f  Muscimol d 0.5')
a5.set_ylim(-80, 5)

h = [Line2D([], [], color=style.ARCH[a], lw=1.4, ls=LS[a], label=NAME[a]) for a in ARCHS]
h += [Line2D([], [], color=style.MUTED, lw=0, marker='D', ms=3, mfc='white', label='hand-built A / B / C'),
      plt.Rectangle((0, 0), 1, 1, color=style.MUTED, alpha=0.2, label='trained: median line, network range shaded')]
leg.legend(handles=h, loc='upper left', fontsize=6, borderaxespad=0, handlelength=2.2)
leg.text(0, 0.30, 'RT curves stop where any network of an architecture has too few\ncompleted saccades into the '
         'injected field (A, AE, C and B at d 0.35);\nmuscimol RT undefined for the same reason in 10 of 24 '
         'networks.\nAll architectures show the same phenotypes (no resolver vs hub difference);\n'
         'B is weaker (RT -15 to -21 ms at d 0.2; fewer fixation breaks).', fontsize=5.6, color=style.MUTED,
         transform=leg.transAxes, va='top')
style.save(fig, str(HERE / 'D1_bicuculline_dose_response'))
