#!/usr/bin/env python3
"""
Build results/aim1_panels/D1_bicuculline_dose_response/ from results/hub_models/bicuculline/per_network.csv
(scripts/srs_bicuculline.py + srs_bicuculline_summary.py; per-network means over 3 seeds; left SC = right field is
injected). No simulation. Checks each D1 write-up number against the table (stats.csv).
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'aim1_panels' / 'D1_bicuculline_dose_response'
SRC = ROOT / 'results' / 'hub_models' / 'bicuculline' / 'per_network.csv'
DOSE = {'ctrl': 0.0, 'full05': 0.05, 'full10': 0.1, 'full20': 0.2, 'full35': 0.35}
MEAS = {'drt_inj': 'rt_change_injected_field_ms', 'drt_oth': 'rt_change_other_field_ms',
        'fixbreak': 'fixation_break_rate', 'fixbreak_right': 'fixation_break_fraction_toward_injected',
        'dp_right': 'delta_p_choose_injected_field', 'db0': 'delta_logit_bias_toward_injected_value_controlled'}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(SRC)
    m['handbuilt'] = m['net'].str.startswith('hand_')
    m['arch'] = m['group'].str.replace('hand_', '', regex=False)
    keep = m[m['cond'].isin(list(DOSE) + ['musc50'])].copy()
    keep['manipulation'] = np.where(keep['cond'] == 'musc50', 'muscimol', np.where(keep['cond'] == 'ctrl', 'control',
                                                                                     'bicuculline (full)'))
    keep['dose_d'] = keep['cond'].map(DOSE).fillna(0.5)
    long = keep.melt(id_vars=['net', 'arch', 'handbuilt', 'cond', 'manipulation', 'dose_d'], value_vars=list(MEAS),
                     var_name='source_column', value_name='value')
    long['measure'] = long['source_column'].map(MEAS)
    long = long.sort_values(['measure', 'handbuilt', 'arch', 'net', 'dose_d'])
    long.to_csv(OUT / 'D1_per_network_long.csv', index=False)

    # architecture summary (trained networks only): median and range over networks; plotted only where every
    # network of the architecture has a defined value (truncation rule for RT, which is undefined when < 21
    # completed single-target saccades into the field remain)
    rows = []
    tr = long[~long['handbuilt']]
    for (meas, arch, cond), g in tr.groupby(['measure', 'arch', 'cond']):
        v = g['value'].dropna()
        rows.append(dict(measure=meas, arch=arch, cond=cond, manipulation=g['manipulation'].iloc[0],
                         dose_d=g['dose_d'].iloc[0], n_networks=len(g), n_defined=len(v),
                         median=v.median() if len(v) else np.nan, min=v.min() if len(v) else np.nan,
                         max=v.max() if len(v) else np.nan, plotted=len(v) == len(g)))
    summ = pd.DataFrame(rows).sort_values(['measure', 'arch', 'dose_d'])
    summ.to_csv(OUT / 'D1_architecture_summary.csv', index=False)

    # ---- write-up checks
    S = []
    def add(stat, value, n, test, note=''):
        S.append(dict(statistic=stat, value=value, n=n, test=test, source=str(SRC.relative_to(ROOT)), note=note))
    w = keep.set_index(['net', 'cond'])
    nets = keep['net'].unique()
    add('networks', len(nets), len(nets), 'count', f'trained {(~keep.drop_duplicates("net").handbuilt).sum()}, '
        f'hand-built {keep.drop_duplicates("net").handbuilt.sum()}')
    d20 = keep[keep.cond == 'full20']
    t20 = d20[~d20.handbuilt]
    add('d 0.2 RT change into injected field (ms), trained networks min..max', f'{t20.drt_inj.max():.1f} .. '
        f'{t20.drt_inj.min():.1f}', len(t20), 'per-network seed mean', 'write-up "17-42 ms faster"')
    med = t20.groupby('arch').drt_inj.median()
    add('d 0.2 RT change into injected field, architecture medians (ms)', '; '.join(f'{a} {v:.1f}' for a, v in
        med.items()), len(t20), 'median over networks', '17-42 = range of architecture MEDIANS')
    for a, g in t20.groupby('arch'):
        add(f'd 0.2 RT change injected field, {a} range (ms)', f'{g.drt_inj.max():.1f} .. {g.drt_inj.min():.1f}', len(g),
            'min..max over networks')
    h20 = d20[d20.handbuilt]
    add('d 0.2 RT change injected field, hand-built (ms)', '; '.join(f'{r.net} {r.drt_inj:.1f}' for r in
        h20.itertuples()), len(h20), 'single network')
    add('d 0.2 RT change OTHER field, all networks min..max (ms)', f'{d20.drt_oth.min():.1f} .. {d20.drt_oth.max():.1f}',
        len(d20), 'per-network', 'largest |change|: ' + d20.loc[d20.drt_oth.abs().idxmax(), 'net'])
    d35 = keep[keep.cond == 'full35']
    add('d 0.35 fixation-break rate, all 24 networks min..max', f'{d35.fixbreak.min():.3f} .. {d35.fixbreak.max():.3f}',
        len(d35), 'fraction of all trials', 'below 0.20: ' + ', '.join(f'{r.net} {r.fixbreak:.3f}' for r in
                                                                        d35[d35.fixbreak < 0.2].itertuples()))
    t35 = d35[~d35.handbuilt]
    add('d 0.35 fixation-break rate, trained networks min..max', f'{t35.fixbreak.min():.3f} .. {t35.fixbreak.max():.3f}',
        len(t35), 'fraction of all trials')
    fr = d35.fixbreak_right.dropna()
    add('d 0.35 fraction of fixation breaks toward injected field, min..max (defined only)',
        f'{fr.min():.3f} .. {fr.max():.3f}', len(fr), 'fraction', f'undefined (<= 10 breaks) in '
        f'{d35.fixbreak_right.isna().sum()} networks: ' + ', '.join(d35[d35.fixbreak_right.isna()].net))
    for c in ('full35',):
        g = keep[keep.cond == c]
        add(f'{c}: networks with RT into injected field undefined', int(g.drt_inj.isna().sum()), len(g), 'count',
            ', '.join(g[g.drt_inj.isna()].net))
    for c in ('full05', 'full10', 'full20', 'full35'):
        g = keep[keep.cond == c]
        neg = g[g.dp_right < 0]
        add(f'{c}: networks with P(choose injected) DEcreasing', len(neg), len(g), 'sign count',
            ', '.join(f'{r.net} {r.dp_right:+.3f}' for r in neg.itertuples()))
        negb = g[g.db0 < 0]
        add(f'{c}: networks with logit bias b0 (value-controlled) moving AWAY', len(negb), len(g), 'sign count',
            ', '.join(f'{r.net} {r.db0:+.2f}' for r in negb.itertuples()))
    for c in ('full05', 'full10'):
        g = keep[(keep.cond == c) & (keep.arch == 'C') & ~keep.handbuilt]
        add(f'{c}: C db0 (logit bias toward injected), per network', '; '.join(f'{r.net} {r.db0:+.2f}' for r in
            g.itertuples()), len(g), 'per-network', 'negative = away; dp_right for the same: ' +
            '; '.join(f'{r.net} {r.dp_right:+.3f}' for r in g.itertuples()))
    mu = keep[keep.cond == 'musc50']
    v = mu.drt_inj.dropna()
    add('muscimol d 0.5 RT change into injected field (ms), defined networks min..max', f'+{v.min():.0f} .. +{v.max():.0f}',
        len(v), 'per-network', f'undefined (fewer than 21 completed saccades into field) in {mu.drt_inj.isna().sum()}'
        ' networks: ' + ', '.join(mu[mu.drt_inj.isna()].net))
    tm = mu[~mu.handbuilt].groupby('arch').drt_inj.median()
    add('muscimol RT change, architecture medians (ms)', '; '.join(f'{a} {x:+.0f}' for a, x in tm.items()), len(mu),
        'median over defined networks', 'write-up "+120-210 ms" ~ range of architecture medians')
    add('muscimol delta P(choose injected), min..max', f'{mu.dp_right.min():+.3f} .. {mu.dp_right.max():+.3f}', len(mu),
        'per-network', 'all negative = away' if (mu.dp_right < 0).all() else 'not all negative')
    add('control fixation-break rate, max', keep[keep.cond == 'ctrl'].fixbreak.max(), len(nets), 'per-network',
        keep.loc[keep[keep.cond == 'ctrl'].fixbreak.idxmax(), 'net'])
    pd.DataFrame(S).to_csv(OUT / 'stats.csv', index=False)
    shutil.copy(HERE / 'style.py', OUT / 'style.py')
    pd.set_option('display.max_colwidth', 300, 'display.width', 300)
    print(pd.DataFrame(S)[['statistic', 'value', 'n', 'note']].to_string())


if __name__ == '__main__':
    main()
