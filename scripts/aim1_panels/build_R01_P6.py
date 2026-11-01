#!/usr/bin/env python3
"""
R01 core P6: model-to-monkey comparison of ipsilateral activation (conflict-task-RNN Prediction 2).

Model (conflict-task-RNN, task-trained bilateral SC E-I networks on the Conflict Task; read-only source
`conflict-task-RNN/figures/probe_main_tables.md` and `probe_commEI_tables.md`, written by code/probe.py): fraction of
units activated (paired Wilcoxon vs -100..0 ms baseline, p < 0.01, 50-150 ms) by a single target at ANY contralateral /
ANY ipsilateral location (4 each), mean +- SD over networks.
Monkey (4factors unilateral recordings, `results/intercollicular/g1_geometry/rows.npy` from scripts/ic_g1_geometry.py,
diagonal-target sessions, meridian session excluded): activated (paired t vs pre-target baseline, p < 0.01, visual
window) at any contralateral location tested (P or contra-neighbour) / any ipsilateral location tested (ipsi-neighbour or
opposite); Wilson 95% CI. Classic neurons <-> model E units; putative interneurons <-> model I units.
Output: results/aim1_panels/R01_core/P6_model_vs_monkey_ipsilateral/
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

ROOT = Path(__file__).parent.parent.parent
OUT = ROOT / 'results' / 'aim1_panels' / 'R01_core' / 'P6_model_vs_monkey_ipsilateral'
A = 0.01

# model values (mean, SD over networks) from conflict-task-RNN probe tables (visual window classification)
MODEL = [
    # group, n_nets, class, act_contra (mean, sd), act_ipsi (mean, sd)
    ('trained (bilateral)', 10, 'E', (1.00, 0.00), (0.00, 0.00)),
    ('trained (bilateral)', 10, 'I commissural', (0.62, 0.11), (0.02, 0.02)),
    ('trained (bilateral)', 10, 'I local', (0.45, 0.10), (0.04, 0.04)),
    ('trained + cross-midline E->I', 5, 'E', (1.00, 0.00), (0.00, 0.00)),
    ('trained + cross-midline E->I', 5, 'I commissural', (0.63, 0.10), (0.16, 0.05)),
    ('trained + cross-midline E->I', 5, 'I local', (0.54, 0.06), (0.07, 0.02)),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [r for r in np.load(ROOT / 'results/intercollicular/g1_geometry/rows.npy', allow_pickle=True)
            if not r['meridian']]
    act = lambda r, k: (r.get(f'visual_{k}_p', np.nan) < A) and (r.get(f'visual_{k}', 0) > 0)
    data = []
    for monkey in ('both', 'Feynman', 'Newton'):
        for typ, lab in (('E', 'classic'), ('I', 'putative interneuron')):
            rr = [r for r in rows if r['type'] == typ and (monkey == 'both' or r['monkey'] == monkey)]
            n = len(rr)
            for side, keys in (('contra', ('P', 'cn')), ('ipsi', ('in', 'op')), ('ipsi_neighbour_only', ('in',))):
                k = sum(any(act(r, kk) for kk in keys) for r in rr)
                lo, hi = proportion_confint(k, n, method='wilson')
                data.append(dict(source='monkey', monkey=monkey, unit_class=lab, side=side, n_units=n, n_activated=k,
                                 frac=k / n, ci95_lo=lo, ci95_hi=hi))
    model = []
    for g, nn, cl, c, i in MODEL:
        for side, (m, sd) in (('contra', c), ('ipsi', i)):
            model.append(dict(source='model', group=g, n_networks=nn, unit_class=cl, side=side, frac_mean=m, frac_sd=sd))
    pd.DataFrame(data).to_csv(OUT / 'P6_monkey.csv', index=False)
    pd.DataFrame(model).to_csv(OUT / 'P6_model.csv', index=False)
    shutil.copy(ROOT / 'scripts/aim1_panels/style.py', OUT / 'style.py')
    print(pd.DataFrame(data).round(3).to_string())


if __name__ == '__main__':
    main()
