#!/usr/bin/env python3
"""
Build results/aim1_panels/C1_rt_value_signature/: S9 (Spearman single-target RT vs value of the shown identity) for
every task-trained and hand-built srsClassic network at full engagement (w = 1, trained threshold), against the
recorded 08-20 (and 08-19) value; robustness to RT shifts; binned RT vs value curves (normalised RT).

Simulation (cheap, ~4 s per network x seed on one CPU thread): exactly the code path of scripts/srs_rt_shift_pilot.py
at theta offset 0 (srs_bicuculline.tables / run, seeds 0, 1; run seed 1000*seed + 7), for all 21 trained v2 networks
(A, AE, B, C retained seeds) and the 3 hand-built networks (srs_bicuculline.HAND parameters). For B and C the result
is checked to equal results/hub_models/rt_shift_pilot/runs.csv (offset 0) exactly.

Usage: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 python scripts/aim1_panels/build_C1.py [--resim]
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu, norm

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'aim1_panels' / 'C1_rt_value_signature'
HM = ROOT / 'results' / 'hub_models'
BELIEF = HM / 'belief'
TRAINED_NETS = [f'{a}_s{s}' for a in ('A', 'AE', 'B') for s in range(6)] + ['C_s0', 'C_s4', 'C_s5']
HAND_NETS = ['hand_A', 'hand_B', 'hand_C']
N_BOOT = 10000
N_BINS = 5


# ----------------------------------------------------------------------------------------------- simulation
def simulate():
    sys.path.insert(0, str(ROOT / 'scripts'))
    import torch
    torch.set_num_threads(1)
    import srs_bicuculline as sb
    from srs import SRSConfig, build_handbuilt
    from srs.task import from_recorded
    n_rec = len(from_recorded(str(sb.BELIEF))['V1'])       # first n_rec rows of each table = recorded 08-20 sequence
    runs, trials = [], []
    for name in TRAINED_NETS + HAND_NETS:
        if name.startswith('hand_'):
            a = name.split('_')[1]
            net = sb.build_handbuilt(a, SRSConfig(device='cpu', shape_vis=sb.SHAPE_VIS), **sb.HAND[a])
            arch = a
        else:
            net = sb.load_net(str(sb.TRAINED / name / 'model.pt'))
            arch = name.split('_')[0]
        for seed in range(2):
            tab = sb.tables(net.cfg, seed)
            r = sb.run(net, tab, seed=1000 * seed + 7)
            st = tab['trial_type'] == 0
            sid = np.where(tab['id_R'] > 0, tab['id_R'], tab['id_L'])
            loc = np.where(tab['id_R'] > 0, 1, 0)
            v = np.where(sid == 1, tab['V1'], tab['V2'])
            ok = st & (r['outcome'] == 0) & np.isfinite(r['rt'])
            rec = np.arange(len(v)) < n_rec
            okr, oks = ok & rec, ok & ~rec
            runs.append(dict(net=name, arch=arch, handbuilt=name.startswith('hand_'), seed=seed,
                             n_single_completed=int(ok.sum()), median_rt_ms=float(np.median(r['rt'][ok])),
                             s9_rho=spearmanr(v[ok], r['rt'][ok])[0],
                             n_single_completed_recseq=int(okr.sum()),
                             s9_rho_recseq=spearmanr(v[okr], r['rt'][okr])[0],
                             s9_rho_synthetic=spearmanr(v[oks], r['rt'][oks])[0]))
            idx = np.where(ok)[0]
            trials.append(pd.DataFrame(dict(net=name, arch=arch, handbuilt=name.startswith('hand_'), seed=seed,
                                            trial_row=idx,
                                            sequence=np.where(rec[idx], 'recorded_08-20', 'synthetic'),
                                            target_field=np.where(loc[idx] == 1, 'right', 'left'),
                                            value_shown_ms=v[idx], rt_ms=r['rt'][idx])))
        print(name, 'done', flush=True)
    return pd.DataFrame(runs), pd.concat(trials, ignore_index=True)


# ----------------------------------------------------------------------------------------------- recorded
def recorded(ses):
    """Single-target trials exactly as scripts/srs_belief_fit.py (S9): correct single-target, RT >= 80 ms; value =
    delta-rule belief (ms) of the shown identity before the trial; block-mean = true block mean reward of that identity."""
    d = pd.read_csv(BELIEF / f'{ses}_trials_with_belief.csv')
    st = ((d['n_targets'] == 1) & d['outcome'].isin(['SINGLE_T1_CORRECT', 'SINGLE_T2_CORRECT'])
          & (d['rt_s'] >= 0.08)).to_numpy()
    t1 = (d['outcome'] == 'SINGLE_T1_CORRECT').to_numpy()
    out = pd.DataFrame(dict(session=ses, trial_row=d['trial_row'], block=d['conflict_block'],
                            shown_identity=np.where(t1, 1, 2),
                            value_belief_ms=np.where(t1, d['belief_t1_ms'], d['belief_t2_ms']),
                            value_blockmean_ms=np.where(t1, d['block_mean_t1_ms'], d['block_mean_t2_ms']),
                            rt_ms=d['rt_s'] * 1000))[st].reset_index(drop=True)
    out['trial_order'] = np.arange(len(out))
    return out


def rho_ci(x, y, rng):
    r, p = spearmanr(x, y)
    n = len(x)
    z, se = np.arctanh(r), np.sqrt(1.06 / (n - 3))           # Fieller, Hartley & Pearson (1957) SE for Spearman
    f_lo, f_hi = np.tanh(z + np.array([-1, 1]) * norm.ppf(0.975) * se)
    bs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        i = rng.integers(0, n, n)
        bs[b] = spearmanr(x[i], y[i])[0]
    b_lo, b_hi = np.percentile(bs, [2.5, 97.5])
    return dict(rho=r, p=p, n=n, fisher_lo=f_lo, fisher_hi=f_hi, boot_lo=b_lo, boot_hi=b_hi,
                se_fisher=se, se_boot=bs.std(ddof=1))


def partial_spearman(x, y, z):
    """Spearman partial correlation of x, y controlling for z (rank-transform, residualise linearly)."""
    from scipy.stats import rankdata, pearsonr
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    A = np.c_[np.ones_like(rz), rz]
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return pearsonr(ex, ey)


def binned(x, rt, edges, rng, label):
    """Mean % RT change from the source's mean single-target RT per value bin, bootstrap 95% CI (trials resampled
    within bin)."""
    pct = 100 * (rt - rt.mean()) / rt.mean()
    rows = []
    b = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    for k in range(len(edges) - 1):
        m = b == k
        if m.sum() < 5:
            continue
        y = pct[m]
        bs = np.array([y[rng.integers(0, len(y), len(y))].mean() for _ in range(2000)])
        rows.append(dict(**label, bin=k, value_bin_lo_ms=edges[k], value_bin_hi_ms=edges[k + 1],
                         value_mean_ms=x[m].mean(), n_trials=int(m.sum()), rt_mean_ms=rt[m].mean(),
                         rt_pct_change_from_mean=y.mean(), ci95_lo=np.percentile(bs, 2.5),
                         ci95_hi=np.percentile(bs, 97.5), source_mean_rt_ms=rt.mean()))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261101)
    sim_runs, sim_trials = OUT / 'C1_sim_runs.csv', OUT / 'C1_model_single_target_trials.csv'
    if '--resim' in sys.argv or not sim_runs.exists():
        runs, trials = simulate()
        runs.to_csv(sim_runs, index=False)
        trials.to_csv(sim_trials, index=False)
    runs, trials = pd.read_csv(sim_runs), pd.read_csv(sim_trials)

    # check against the RT-shift pilot (offset 0) for B and C
    pil = pd.read_csv(HM / 'rt_shift_pilot' / 'runs.csv')
    p0 = pil[pil['off'] == 0].set_index(['net', 'seed'])['s9']
    chk = runs.set_index(['net', 'seed'])['s9_rho'].reindex(p0.index)
    maxdiff = float(np.nanmax(np.abs(chk - p0)))
    print('max |S9 - rt_shift_pilot offset 0| =', maxdiff)
    assert maxdiff < 1e-9, 'simulation does not reproduce rt_shift_pilot/runs.csv'

    # ---- panel a: per-network S9 (mean of 2 seeds)
    a = runs.groupby(['net', 'arch', 'handbuilt'], as_index=False).agg(
        s9_rho=('s9_rho', 'mean'), s9_rho_seed0=('s9_rho', 'first'), s9_rho_seed1=('s9_rho', 'last'),
        n_single_completed_per_seed=('n_single_completed', 'mean'), median_rt_ms=('median_rt_ms', 'mean'),
        s9_rho_recseq=('s9_rho_recseq', 'mean'), s9_rho_synthetic=('s9_rho_synthetic', 'mean'))
    bic = pd.read_csv(HM / 'bicuculline' / 'per_network.csv')
    ctrl = bic[bic['cond'] == 'ctrl'].set_index('net')
    a['bic_ctrl_rho_inj_rightfield'] = a['net'].map(ctrl['rho_inj'])
    a['bic_ctrl_rho_oth_leftfield'] = a['net'].map(ctrl['rho_oth'])
    order = {'A': 0, 'AE': 1, 'B': 2, 'C': 3}
    a['arch_x'] = a['arch'].map(order)
    a['role'] = a['arch'].map({'A': 'resolver', 'AE': 'resolver (input onto E only)', 'B': 'priority hub',
                               'C': 'outcome hub'})
    a = a.sort_values(['arch_x', 'handbuilt', 'net']).reset_index(drop=True)
    a.to_csv(OUT / 'C1_ab_s9_per_network.csv', index=False)

    rec = {s: recorded(s) for s in ('08-20', '08-19')}
    band = []
    for s, d in rec.items():
        c = rho_ci(d['value_belief_ms'].to_numpy(), d['rt_ms'].to_numpy(), rng)
        band.append(dict(session=s, value='belief', **c))
        c = rho_ci(d['value_blockmean_ms'].to_numpy(), d['rt_ms'].to_numpy(), rng)
        band.append(dict(session=s, value='block_mean', **c))
    band = pd.DataFrame(band)
    band.to_csv(OUT / 'C1_ab_recorded_s9.csv', index=False)

    # ---- panel c: robustness to the readout threshold (B, C from the pilot), per-network seed means
    rb = pil.groupby(['net', 'arch', 'off'], as_index=False).agg(s9_rho=('s9', 'mean'), median_rt_ms=('rt', 'mean'))
    base = rb[rb['off'] == 0].set_index('net')
    rb['delta_rt_ms_from_offset0'] = rb['median_rt_ms'] - rb['net'].map(base['median_rt_ms'])
    rb['delta_s9_from_offset0'] = rb['s9_rho'] - rb['net'].map(base['s9_rho'])
    rb = rb.rename(columns={'off': 'theta_offset'})
    rb.to_csv(OUT / 'C1_c_rt_shift_robustness.csv', index=False)

    # ---- representative networks: closest to their architecture's median S9 (full tables)
    m20 = rec['08-20']
    tr = trials[~trials['handbuilt']]
    med = a[~a['handbuilt']].groupby('arch')['s9_rho'].median()
    rep = {}
    for arch in ('A', 'B', 'C'):
        g = a[(a['arch'] == arch) & ~a['handbuilt']]
        rep[arch] = g.loc[(g['s9_rho'] - med[arch]).abs().idxmin(), 'net']

    # ---- panel d: RT vs the TRUE value of the shown identity. Monkey: true block-mean reward, quartile bins
    # (16 block x identity cells, so bins hold whole cells). Models: synthetic sessions (value input = delta-rule belief, alpha 0.1, close to block
    # means), binned at each network's value quintiles.
    xb = m20['value_blockmean_ms'].to_numpy()
    eb = np.quantile(xb, np.linspace(0, 1, 5)); eb[0], eb[-1] = -np.inf, np.inf
    rows = binned(xb, m20['rt_ms'].to_numpy(), eb, rng,
                  dict(source='monkey 08-20', arch='data', net='Newton 08-20', value_type='block-mean reward'))
    for arch, net in rep.items():
        g = tr[(tr['net'] == net) & (tr['sequence'] == 'synthetic')]
        x = g['value_shown_ms'].to_numpy()
        e = np.quantile(x, np.linspace(0, 1, N_BINS + 1)); e[0], e[-1] = -np.inf, np.inf
        rows += binned(x, g['rt_ms'].to_numpy(), e, rng,
                       dict(source='model', arch=arch, net=net, value_type='belief input, synthetic sessions'))
    pd.DataFrame(rows).to_csv(OUT / 'C1_d_rt_vs_true_value.csv', index=False)

    # ---- panel e: RT vs the fitted belief on the 08-20 sequence (the recorded S9 variable). Same bin edges
    # (08-20 belief quintiles) for monkey and models; models were given exactly these beliefs as value input.
    edges = np.quantile(m20['value_belief_ms'], np.linspace(0, 1, N_BINS + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    rows = binned(m20['value_belief_ms'].to_numpy(), m20['rt_ms'].to_numpy(), edges, rng,
                  dict(source='monkey 08-20', arch='data', net='Newton 08-20', value_type='delta-rule belief'))
    for arch, net in rep.items():
        g = tr[(tr['net'] == net) & (tr['sequence'] == 'recorded_08-20')]
        rows += binned(g['value_shown_ms'].to_numpy(), g['rt_ms'].to_numpy(), edges, rng,
                       dict(source='model', arch=arch, net=net, value_type='delta-rule belief (same 08-20 sequence)'))
    # monkey, drift-controlled: RT relative to its own block's mean RT (between-block drift removed)
    blk_rt = m20.groupby('block')['rt_ms'].transform('mean')
    rows += binned(m20['value_belief_ms'].to_numpy(), (m20['rt_ms'] - blk_rt + m20['rt_ms'].mean()).to_numpy(), edges,
                   rng, dict(source='monkey 08-20', arch='data_within_block', net='Newton 08-20',
                             value_type='delta-rule belief; RT minus own block mean RT'))
    pd.DataFrame(rows).to_csv(OUT / 'C1_e_rt_vs_belief_0820_sequence.csv', index=False)
    rep_rho = []
    for arch, net in rep.items():
        for sq in ('recorded_08-20', 'synthetic'):
            g = tr[(tr['net'] == net) & (tr['sequence'] == sq)]
            r, p = spearmanr(g['value_shown_ms'], g['rt_ms'])
            rep_rho.append(dict(arch=arch, net=net, seq=sq, n=len(g), rho=r, p=p))
    rep_rho = pd.DataFrame(rep_rho)
    # within-block rich vs poor identity RT difference (monkey; true value contrast free of slow drift)
    wb = []
    for blk, g in m20.groupby('block'):
        hi = g['value_blockmean_ms'] == g['value_blockmean_ms'].max()
        wb.append(100 * (g.loc[hi, 'rt_ms'].mean() - g.loc[~hi, 'rt_ms'].mean()) / g['rt_ms'].mean())
    wb = np.array(wb)
    # within-block rich - lean RT in ms, bootstrap over trials stratified by block
    def rich_lean_ms(d):
        out = []
        for _, g in d.groupby('block'):
            hi = g['value_blockmean_ms'] == g['value_blockmean_ms'].max()
            out.append(g.loc[hi, 'rt_ms'].mean() - g.loc[~hi, 'rt_ms'].mean())
        return np.mean(out)
    wb_ms = rich_lean_ms(m20)
    bsw = []
    for _ in range(2000):
        bsw.append(rich_lean_ms(m20.groupby('block', group_keys=False).apply(
            lambda g: g.iloc[rng.integers(0, len(g), len(g))])))
    wb_ci = np.percentile(bsw, [2.5, 97.5])
    import statsmodels.formula.api as smf
    fe = smf.ols('np.log(rt_ms) ~ value_belief_ms + C(block)', data=m20).fit()
    r_bt = spearmanr(m20['trial_order'], m20['value_belief_ms'])

    # ---- stats
    S = []
    def add(stat, value, n, test, source, note=''):
        S.append(dict(statistic=stat, value=value, n=n, test=test, source=source, note=note))
    src_sim = 'scripts/aim1_panels/build_C1.py simulation (= srs_rt_shift_pilot.py code, offset 0, seeds 0-1)'
    for s in ('08-20', '08-19'):
        r = band[(band.session == s) & (band.value == 'belief')].iloc[0]
        add(f'recorded S9 {s} (belief value)', r.rho, r.n, 'Spearman; p two-sided', 'results/hub_models/belief/'
            f'{s}_trials_with_belief.csv (filter as scripts/srs_belief_fit.py)', f'p = {r.p:.2g}')
        add(f'recorded S9 {s} 95% CI (bootstrap percentile, {N_BOOT} resamples)', f'[{r.boot_lo:.3f}, {r.boot_hi:.3f}]',
            r.n, 'bootstrap over trials', 'same')
        add(f'recorded S9 {s} 95% CI (Fisher z, SE sqrt(1.06/(n-3)))', f'[{r.fisher_lo:.3f}, {r.fisher_hi:.3f}]', r.n,
            'Fieller-Hartley-Pearson', 'same')
        r = band[(band.session == s) & (band.value == 'block_mean')].iloc[0]
        add(f'recorded S9 {s} with block-mean value (drift control)', r.rho, r.n, 'Spearman', 'same',
            f'p = {r.p:.2g}; bootstrap CI [{r.boot_lo:.3f}, {r.boot_hi:.3f}]')
    pr = partial_spearman(m20['value_belief_ms'], m20['rt_ms'], m20['trial_order'])
    add('recorded S9 08-20 partial on trial order (drift control)', pr[0], len(m20), 'partial Spearman', 'same',
        f'p = {pr[1]:.2g}')
    r_drift = spearmanr(m20['trial_order'], m20['rt_ms'])
    add('08-20 RT vs trial order', r_drift[0], len(m20), 'Spearman', 'same', f'p = {r_drift[1]:.2g}')
    add('08-20 mean single-target RT (ms)', m20['rt_ms'].mean(), len(m20), 'mean', 'same',
        f'median {m20["rt_ms"].median():.1f}')
    for arch in ('A', 'AE', 'B', 'C'):
        g = a[(a['arch'] == arch) & ~a['handbuilt']]['s9_rho']
        add(f'S9 trained {arch} median [min, max], w = 1, offset 0', f'{g.median():+.3f} [{g.min():+.3f}, {g.max():+.3f}]',
            len(g), 'per-network mean of 2 seeds', src_sim)
    for net in HAND_NETS:
        g = a[a['net'] == net].iloc[0]
        add(f'S9 {net}, w = 1', g.s9_rho, 2, 'mean of 2 seeds', src_sim)
    st0 = pd.read_csv(HM / 'stage0' / 'draws.csv').groupby('arch')['S9_rt_value_rho'].agg(['mean', 'std', 'count'])
    for arch, r in st0.iterrows():
        add(f'S9 hand-built {arch}, CALIBRATED engagement (stage0; A,B w 0.3, C w 0.1)', f'{r["mean"]:+.3f} +- {r["std"]:.3f}',
            int(r['count']), 'mean +- SD over draws', 'results/hub_models/stage0/draws.csv',
            'these are the write-up hand-built values (-0.32 / +0.02); not full engagement')
    gB = a[(a.arch == 'B') & ~a.handbuilt]['s9_rho']
    gC = a[(a.arch == 'C') & ~a.handbuilt]['s9_rho']
    gR = a[a.arch.isin(['A', 'AE']) & ~a.handbuilt]['s9_rho']
    u = mannwhitneyu(gB, gC, alternative='two-sided', method='exact')
    add('B vs C S9 rank-sum', u.pvalue, '6 vs 3', 'Mann-Whitney U exact two-sided', src_sim, f'U = {u.statistic}')
    u = mannwhitneyu(gR, gC, alternative='two-sided', method='exact')
    add('resolvers (A+AE) vs C S9 rank-sum', u.pvalue, '12 vs 3', 'Mann-Whitney U exact two-sided', src_sim,
        f'U = {u.statistic}')
    u = mannwhitneyu(gR, gB, alternative='two-sided', method='exact')
    add('resolvers (A+AE) vs B S9 rank-sum', u.pvalue, '12 vs 6', 'Mann-Whitney U exact two-sided', src_sim,
        f'U = {u.statistic}')
    r20 = band[(band.session == '08-20') & (band.value == 'belief')].iloc[0]
    se = 1 / np.sqrt(r20.n - 3)
    add('08-20 gap to most negative C network, offset 0, in SE (SE = 1/sqrt(n-3))', (gC.min() - r20.rho) / se, r20.n,
        'z distance', src_sim, f'most negative C at offset 0 = {gC.min():+.3f}')
    cmin_all = rb[rb.arch == 'C']['s9_rho'].min()
    add('08-20 gap to most negative C network over all offsets, in SE', (cmin_all - r20.rho) / se, r20.n, 'z distance',
        'results/hub_models/rt_shift_pilot/runs.csv', f'most negative C over offsets = {cmin_all:+.3f}')
    for arch in ('B', 'C'):
        g = rb[rb.arch == arch]
        add(f'{arch} S9 range over all offsets 0-0.5', f'[{g.s9_rho.min():+.3f}, {g.s9_rho.max():+.3f}]', g.net.nunique(),
            'per-network seed means', 'results/hub_models/rt_shift_pilot/runs.csv')
        g5 = g[g.theta_offset == 0.5]
        add(f'{arch} RT slowing at offset 0.5 (ms), range over networks',
            f'[{g5.delta_rt_ms_from_offset0.min():.1f}, {g5.delta_rt_ms_from_offset0.max():.1f}]', len(g5),
            'median RT difference', 'results/hub_models/rt_shift_pilot/runs.csv')
        add(f'{arch} max |delta S9| over offsets', g.delta_s9_from_offset0.abs().max(), g.net.nunique(),
            'per-network', 'results/hub_models/rt_shift_pilot/runs.csv',
            'network ' + g.loc[g.delta_s9_from_offset0.abs().idxmax(), 'net'])
    up = pil[pil.off == 0].groupby('arch')['dis_rho'].median()
    for arch, v in up.items():
        add(f'{arch} upstream (disinhibition) vs value rho, single-target, median over networks', v,
            pil[(pil.off == 0) & (pil.arch == arch)].net.nunique(), 'Spearman', 'rt_shift_pilot/runs.csv')
    chkd = (a['s9_rho'] - (a['bic_ctrl_rho_inj_rightfield'])).abs()
    add('agreement check: per-network S9 (both fields) vs bicuculline ctrl rho_inj (right field, 3 seeds), Spearman',
        spearmanr(a['s9_rho'], a['bic_ctrl_rho_inj_rightfield'])[0], len(a), 'Spearman across networks',
        'results/hub_models/bicuculline/per_network.csv', 'rho_inj = same statistic restricted to targets in the '
        f'right field; median |difference| {chkd.median():.3f}')
    for _, r in rep_rho.iterrows():
        add(f'representative {r.arch} network ({r.net}) S9, {r.seq} trials only', r.rho, r.n, 'Spearman',
            src_sim, f'p = {r.p:.2g}; 2 seeds pooled')
    for arch in ('A', 'AE', 'B', 'C'):
        g = a[(a['arch'] == arch) & ~a['handbuilt']]
        add(f'S9 trained {arch} on the recorded 08-20 value sequence only, median [min, max]',
            f'{g.s9_rho_recseq.median():+.3f} [{g.s9_rho_recseq.min():+.3f}, {g.s9_rho_recseq.max():+.3f}]', len(g),
            'per-network mean of 2 seeds (~320 single-target trials per seed)', src_sim)
        add(f'S9 trained {arch} on synthetic sessions only, median [min, max]',
            f'{g.s9_rho_synthetic.median():+.3f} [{g.s9_rho_synthetic.min():+.3f}, {g.s9_rho_synthetic.max():+.3f}]',
            len(g), 'per-network mean of 2 seeds', src_sim)
    for net in HAND_NETS:
        g = a[a['net'] == net].iloc[0]
        add(f'S9 {net} on the recorded 08-20 value sequence only', g.s9_rho_recseq, 2, 'mean of 2 seeds', src_sim)
    from scipy.stats import wilcoxon
    w = wilcoxon(wb)
    add('08-20 within-block RT difference rich - lean identity (ms), mean over 8 blocks', wb_ms, len(m20),
        'bootstrap over trials within block (2000)', 'belief csv', f'95% CI [{wb_ci[0]:+.1f}, {wb_ci[1]:+.1f}] ms; '
        'positive = rich SLOWER; a value effect predicts negative')
    add('08-20 log RT ~ belief + block fixed effects: belief coefficient (per ms)', fe.params['value_belief_ms'],
        len(m20), 'OLS', 'belief csv', f'p = {fe.pvalues["value_belief_ms"]:.2g}')
    add('08-20 belief vs trial order', r_bt[0], len(m20), 'Spearman', 'belief csv', f'p = {r_bt[1]:.2g}')
    add('08-20 within-block RT difference rich - poor identity (% of block mean RT), mean over blocks', wb.mean(),
        len(wb), 'Wilcoxon signed-rank over blocks', 'belief csv', f'p = {w.pvalue:.2g}; per block: '
        + ' '.join(f'{x:+.1f}' for x in wb))
    add('08-20 belief range (shown identity, single-target)', f'[{m20.value_belief_ms.min():.0f}, '
        f'{m20.value_belief_ms.max():.0f}] ms, SD {m20.value_belief_ms.std():.1f}', len(m20), 'range', 'belief csv',
        'delta rule alpha 0.01 (fit at bound)')
    add('08-20 block-mean value range (shown identity)', f'[{m20.value_blockmean_ms.min():.0f}, '
        f'{m20.value_blockmean_ms.max():.0f}] ms, SD {m20.value_blockmean_ms.std():.1f}', len(m20), 'range',
        'belief csv')
    xs = tr[tr.sequence == 'synthetic']['value_shown_ms']
    add('model synthetic-session value range (shown identity)', f'[{xs.min():.0f}, {xs.max():.0f}] ms, SD '
        f'{xs.std():.1f}', len(xs), 'range', src_sim, 'srs.task.make_session: k 1.5, delta-rule alpha 0.1')
    pd.DataFrame(S).to_csv(OUT / 'stats.csv', index=False)
    shutil.copy(HERE / 'style.py', OUT / 'style.py')
    print(pd.DataFrame(S).to_string())
    print('representatives', rep)


if __name__ == '__main__':
    main()
