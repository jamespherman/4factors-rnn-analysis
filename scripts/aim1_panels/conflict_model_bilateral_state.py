#!/usr/bin/env python3
"""Model-to-monkey comparison of the pre-target bilateral SC state (R01 core panel P5).

Trained bilateral SC E-I RNNs of the original Conflict Task (conflict-task-RNN repo; code copied unchanged to
external/conflict_task_rnn/) are simulated with their own noise and trial sampler, and the monkey pre-target analyses
(Newton_SC_2026 code/run_pretarget_bias.m, 2a and 2c; values in results/aim1_panels/R01_core/P4_bilateral_state/) are
applied to the model E units (classic-neuron analogue) and, separately, the I units.

Per network (final_bilat_s0..9, final_nocomm_s0..9):
  * simulate N_TRIALS two-target trials (conflict + congruent, repo sampler) with the network's state noise and input
    noise. Fixation is extended by FIX_EXT ms (t_go and target onset shifted later; everything else unchanged) so the
    300-ms pre-target window never overlaps the start-up transient from x = 0 (native fixation 300-500 ms; monkey
    1.0-1.2 s). The native-timing behavior is simulated too as a check that the extension does not change behavior.
  * window = [target onset - 300 ms, target onset). Primary trial set: dT <= 0 (targets at or before go), so the
    whole window lies inside fixation (fixation input on). Sensitivity: all dT (window then contains go for dT > 0).
  * per unit: mean rate in the window, z-scored across trials; hemisphere population = mean z (E units; I units
    separately). Bias = left SC - right SC (left SC represents the right hemifield, as in the monkey convention).
  * 2a: completed trials; P(chose right) by bias tercile; logistic chose_right ~ z(bias) + big_right + sal_right + dT
    dummies; bias coefficient (log-odds per SD) and Wald z.
  * 2c: all trials in the set; partial correlation of left and right population activity after regressing out
    big_right, sal_right, dT dummies and t_go; p from the t distribution.
  * mechanism controls (paired, same random streams): (i) input noise on the fixation channel removed - the only
    input that reaches both hemispheres; (ii) bilateral only: commissural weights cut post hoc (comm_scale = 0).

CPU, single-threaded. Output: results/aim1_panels/R01_core/P5_model_vs_monkey_bilateral/.
Author: Claude (with James Herman), 2026-11-01.
"""
import copy
import os
import sys
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
import pandas as pd
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'external' / 'conflict_task_rnn'))
from analysis import load_net  # noqa: E402  (copied unchanged from conflict-task-RNN/code)
from task import sample_trials, build_inputs, evaluate_choices, urgency  # noqa: E402
from rollout import valid_step_mask, sample_actions  # noqa: E402

torch.set_num_threads(1)
MODEL_RESULTS = Path(r'D:\OneDrive - University of Pittsburgh\Code\conflict-task-RNN\results')
OUT = ROOT / 'results' / 'aim1_panels' / 'R01_core' / 'P5_model_vs_monkey_bilateral'
OUT.mkdir(parents=True, exist_ok=True)
GROUPS = {'bilateral': [f'final_bilat_s{i}' for i in range(10)],
          'no_commissure': [f'final_nocomm_s{i}' for i in range(10)]}
N_TRIALS = 3500          # ~2000 trials with dT <= 0
CHUNK = 700
FIX_EXT = 500.0          # ms of extra fixation
WIN = 300.0              # ms before target onset
SEED = 20261101


def logistic_wald(y, X):
    """Newton-Raphson logistic regression; X includes intercept. Returns coef, se."""
    b = np.zeros(X.shape[1])
    for _ in range(100):
        p = 1 / (1 + np.exp(-X @ b))
        w = p * (1 - p)
        H = X.T @ (X * w[:, None]) + 1e-9 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        b += step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = 1 / (1 + np.exp(-X @ b))
    H = X.T @ (X * (p * (1 - p))[:, None])
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return b, se


def simulate(net, cfg, n, seed, drop_fix_noise=False):
    """Repo rollout (rollout.run_trials) with the input noise drawn explicitly so conditions can share streams.
    Returns dict of per-trial arrays incl. window-mean rates [n, N_units]."""
    gt, gs, gp = (torch.Generator().manual_seed(seed + k) for k in (0, 1, 2))
    parts = []
    for c0 in range(0, n, CHUNK):
        b = min(CHUNK, n - c0)
        tr = sample_trials(cfg, b, 'cpu', generator=gt)
        u, t = build_inputs(cfg, tr, 'cpu', noise=False)
        eps = cfg.input_noise * torch.randn(u.shape, generator=gt)   # same as build_inputs(noise=True)
        if drop_fix_noise:
            eps[..., 0] = 0.0
        u = u + eps
        with torch.no_grad():
            R = net(u, noise=True, generator=gs)
            urg = urgency(cfg, tr, t, generator=gp)
            logits = net.policy_logits(R, u, urg)
            valid = valid_step_mask(cfg, tr, u.shape[1], 'cpu')
            act_step, act_loc, _ = sample_actions(logits, valid, generator=gp)
            out = evaluate_choices(cfg, tr, act_step, act_loc)
        t_on = (tr['t_go'] + tr['dT'])[:, None]
        inwin = ((t[None, :] >= t_on - WIN) & (t[None, :] < t_on)).float()          # [b, T]
        wmean = (R * inwin[..., None]).sum(1) / inwin.sum(1, keepdim=True)
        # fixation-period trace (from trial start) for the settling check
        parts.append(dict(t_go=tr['t_go'].numpy(), dT=tr['dT'].numpy(), big=tr['big_side'].numpy(),
                          sal=tr['sal_side'].numpy(), conflict=tr['conflict'].numpy(),
                          completed=out['completed'].numpy(), chosen=out['chosen_side'].numpy(),
                          chose_big=out['chose_big'].numpy(), fixbreak=out['fixbreak'].numpy(),
                          win_start=(t_on[:, 0] - WIN).numpy(), wmean=wmean.numpy(),
                          trace=R[:, :int(FIX_EXT / cfg.dt) + 30].mean(0).numpy()[None]))
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != 'trace'}
    d['trace'] = np.mean(np.concatenate([p['trace'] for p in parts]), 0)
    return d


def zpop(W, units):
    """z-score each unit's window rate across trials, mean over units."""
    X = W[:, units]
    sd = X.std(0)
    keep = sd > 1e-9
    Z = (X[:, keep] - X[:, keep].mean(0)) / sd[keep]
    return Z.mean(1), int(keep.sum())


def analyse(d, idx, sel):
    """2a + 2c on the trial subset sel for E and I units."""
    res, terc = {}, []
    dT_levels = np.unique(d['dT'][sel])
    for cls, isc in (('E', idx.is_exc), ('I', idx.is_inh)):
        L, nL = zpop(d['wmean'][sel], np.where(isc & (idx.hemi == 0))[0])
        R, nR = zpop(d['wmean'][sel], np.where(isc & (idx.hemi == 1))[0])
        bias = L - R
        big_r = (d['big'][sel] == 1).astype(float)
        sal_r = (d['sal'][sel] == 1).astype(float)
        dTd = np.stack([(d['dT'][sel] == v).astype(float) for v in dT_levels[1:]], 1) if len(dT_levels) > 1 \
            else np.zeros((sel.sum(), 0))
        # 2c partial correlation (all trials in the set)
        C = np.column_stack([np.ones(sel.sum()), big_r, sal_r, dTd, d['t_go'][sel]])
        rL = L - C @ np.linalg.lstsq(C, L, rcond=None)[0]
        rR = R - C @ np.linalg.lstsq(C, R, rcond=None)[0]
        r, _ = stats.pearsonr(rL, rR)
        dfree = len(rL) - C.shape[1] - 2
        tval = r * np.sqrt(dfree / (1 - r ** 2))
        p_r = 2 * stats.t.sf(abs(tval), dfree)
        r_raw = stats.pearsonr(L, R)[0]
        # 2a choice (completed trials)
        c = d['completed'][sel]
        y = (d['chosen'][sel][c] == 1).astype(float)
        bz = (bias[c] - bias[c].mean()) / bias[c].std()
        X = np.column_stack([np.ones(c.sum()), bz, big_r[c], sal_r[c], dTd[c]])
        coef, se = logistic_wald(y, X)
        # cell-adjusted version: fixed effects for every big x sal x dT cell (the task-determined choice
        # probability); cells where the choice never varies carry no information and are dropped (else the
        # fit separates, as the main-effects fit nearly does: the reward side alone predicts ~all dT <= 0 choices)
        cell = (d['big'][sel][c] * 2 + d['sal'][sel][c]) * 100 + np.searchsorted(dT_levels, d['dT'][sel][c])
        ucell = np.unique(cell)
        mixed = np.array([0 < y[cell == k].mean() < 1 for k in ucell])
        keep = np.isin(cell, ucell[mixed])
        if keep.sum() >= 20 and mixed.sum() >= 1:
            Xc = np.column_stack([(cell[keep] == k).astype(float) for k in ucell[mixed]] +
                                 [(bias[c][keep] - bias[c][keep].mean()) / bias[c][keep].std()])
            cc, sc = logistic_wald(y[keep], Xc)
            cell_coef, cell_z = cc[-1], cc[-1] / sc[-1]
        else:
            cell_coef = cell_z = np.nan
        q = np.quantile(bz, [1 / 3, 2 / 3])
        tt = np.digitize(bz, q)  # 0,1,2
        for k in range(3):
            m = tt == k
            terc.append(dict(unit_class=cls, tercile=k + 1, n_trials=int(m.sum()), p_chose_right=y[m].mean()))
        # bias vs reward side (sanity; trial variables are drawn independently of the pre-target state)
        res[cls] = dict(n_units_left=nL, n_units_right=nR, n_trials_2c=int(sel.sum()), partial_r=r, partial_r_p=p_r,
                        raw_r=r_raw, n_trials_2a=int(c.sum()), bias_coef=coef[1], bias_se=se[1],
                        bias_z=coef[1] / se[1], bias_coef_celladj=cell_coef, bias_z_celladj=cell_z,
                        n_trials_celladj=int(keep.sum()), p_majority_choice=float(np.mean(
                            [max(y[cell == k].mean(), 1 - y[cell == k].mean()) for k in cell])), bigside_coef=coef[2], salside_coef=coef[3],
                        r_bias_bigright=stats.pearsonr(bz, big_r[c])[0])
    return res, terc


def main():
    rows, terc_rows, trace_rows, beh_rows = [], [], [], []
    for group, runs in GROUPS.items():
        for run in runs:
            net, cfg = load_net(MODEL_RESULTS / run, device='cpu')
            cfg_ext = copy.deepcopy(cfg)
            cfg_ext.fix_min += FIX_EXT
            cfg_ext.fix_max += FIX_EXT
            seed = SEED + 1000 * (runs.index(run) + (100 if group == 'no_commissure' else 0))
            conds = {'intact': dict(net_comm=1.0, drop=False),
                     'no_shared_input_noise': dict(net_comm=1.0, drop=True)}
            if group == 'bilateral':
                conds['commissure_cut'] = dict(net_comm=0.0, drop=False)
            for cond, cc in conds.items():
                net.comm_scale.fill_(cc['net_comm'])
                d = simulate(net, cfg_ext, N_TRIALS, seed, drop_fix_noise=cc['drop'])
                sets = {'dT_le_0': d['dT'] <= 0}
                if cond == 'intact':
                    sets['all_dT'] = np.ones(len(d['dT']), bool)
                for sname, sel in sets.items():
                    res, terc = analyse(d, net.idx, sel)
                    for cls, r in res.items():
                        rows.append(dict(group=group, network=run, condition=cond, trial_set=sname,
                                         unit_class=cls, **r))
                    for tr in terc:
                        terc_rows.append(dict(group=group, network=run, condition=cond, trial_set=sname, **tr))
                comp = d['completed']
                beh_rows.append(dict(group=group, network=run, condition=cond, timing=f'fixation +{FIX_EXT:.0f} ms',
                                     completed=comp.mean(), fixbreak=d['fixbreak'].mean(),
                                     p_big_conflict=d['chose_big'][comp & d['conflict']].mean(),
                                     p_big_congruent=d['chose_big'][comp & ~d['conflict']].mean(),
                                     p_right=(d['chosen'][comp] == 1).mean()))
                if cond == 'intact':
                    E = net.idx.is_exc
                    for k, v in enumerate(d['trace']):
                        trace_rows.append(dict(group=group, network=run, t_ms=k * cfg.dt,
                                               mean_rate_E_left=v[E & (net.idx.hemi == 0)].mean(),
                                               mean_rate_E_right=v[E & (net.idx.hemi == 1)].mean()))
            net.comm_scale.fill_(1.0)
            dn = simulate(net, cfg, 2000, seed + 7)   # native timing: behavior check only
            comp = dn['completed']
            beh_rows.append(dict(group=group, network=run, condition='intact', timing='native', completed=comp.mean(),
                                 fixbreak=dn['fixbreak'].mean(),
                                 p_big_conflict=dn['chose_big'][comp & dn['conflict']].mean(),
                                 p_big_congruent=dn['chose_big'][comp & ~dn['conflict']].mean(),
                                 p_right=(dn['chosen'][comp] == 1).mean()))
            last = [r for r in rows if r['network'] == run and r['condition'] == 'intact'
                    and r['trial_set'] == 'dT_le_0' and r['unit_class'] == 'E'][0]
            print(f"{run}: E partial r {last['partial_r']:+.3f} (p {last['partial_r_p']:.2g}), "
                  f"bias coef {last['bias_coef']:+.3f} z {last['bias_z']:+.2f}; cell-adj {last['bias_coef_celladj']:+.3f} "
                  f"z {last['bias_z_celladj']:+.2f} (n {last['n_trials_celladj']})", flush=True)
    per = pd.DataFrame(rows)
    per.to_csv(OUT / 'P5_model_per_network.csv', index=False)
    pd.DataFrame(terc_rows).to_csv(OUT / 'P5_model_tercile_per_network.csv', index=False)
    pd.DataFrame(beh_rows).to_csv(OUT / 'P5_model_behavior_check.csv', index=False)
    pd.DataFrame(trace_rows).to_csv(OUT / 'P5_model_fixation_settling.csv', index=False)
    summarize(per, pd.DataFrame(terc_rows))


def summarize(per, terc):
    P4 = ROOT / 'results' / 'aim1_panels' / 'R01_core' / 'P4_bilateral_state'
    mon_t = pd.read_csv(P4 / 'P4_a_choice_by_bias_tercile.csv')
    mon_s = pd.read_csv(P4 / 'stats.csv')
    # panel CSVs
    t = terc.query("condition == 'intact' and trial_set == 'dT_le_0' and unit_class == 'E'")
    g = t.groupby(['group', 'tercile']).p_chose_right.agg(['mean', 'std', 'count']).reset_index()
    g['sem'] = g['std'] / np.sqrt(g['count'])
    a = pd.concat([mon_t.assign(source='monkey', group=mon_t.session, sd_across_networks=np.nan, n_networks=np.nan,
                                se=mon_t.se_binomial)[['source', 'group', 'tercile', 'p_chose_right', 'se',
                                                       'sd_across_networks', 'n_networks']],
                   g.assign(source='model').rename(columns={'mean': 'p_chose_right', 'sem': 'se',
                                                            'std': 'sd_across_networks', 'count': 'n_networks'})
                   [['source', 'group', 'tercile', 'p_chose_right', 'se', 'sd_across_networks', 'n_networks']]])
    a.to_csv(OUT / 'P5_i_choice_by_bias_tercile.csv', index=False)
    mr = mon_s[mon_s.statistic == 'pretarget_LR_shared_fluctuation_partial_r']
    mb = mon_s[mon_s.statistic == 'pretarget_bias_z']
    pts = [dict(source='monkey', group=r.session, network=r.session, condition='recorded', unit_class='all',
                partial_r=r.value, partial_r_p=r.p, n_trials=r.n) for r in mr.itertuples()]
    ptsb = [dict(source='monkey', group=r.session, network=r.session, condition='recorded', unit_class='all',
                 bias_coef=r.value, bias_z=r.z, n_trials=r.n) for r in mb.itertuples()]
    q = per.query("trial_set == 'dT_le_0'")
    for r in q.itertuples():
        pts.append(dict(source='model', group=r.group, network=r.network, condition=r.condition,
                        unit_class=r.unit_class, partial_r=r.partial_r, partial_r_p=r.partial_r_p,
                        n_trials=r.n_trials_2c))
        ptsb.append(dict(source='model', group=r.group, network=r.network, condition=r.condition,
                         unit_class=r.unit_class, bias_coef=r.bias_coef, bias_z=r.bias_z, n_trials=r.n_trials_2a,
                         bias_coef_celladj=r.bias_coef_celladj, bias_z_celladj=r.bias_z_celladj,
                         n_trials_celladj=r.n_trials_celladj))
    pd.DataFrame(pts).to_csv(OUT / 'P5_ii_LR_partial_correlation.csv', index=False)
    pd.DataFrame(ptsb).to_csv(OUT / 'P5_iii_bias_choice_coefficient.csv', index=False)
    # stats
    srows = []
    for (grp, cond, ts, cls), d in per.groupby(['group', 'condition', 'trial_set', 'unit_class']):
        for stat in ('partial_r', 'raw_r', 'bias_coef', 'bias_z', 'bias_coef_celladj', 'bias_z_celladj',
                     'n_trials_celladj', 'p_majority_choice'):
            v = d[stat].dropna().to_numpy()
            w = stats.wilcoxon(v) if len(v) > 1 and np.any(v != 0) else None
            srows.append(dict(group=grp, condition=cond, trial_set=ts, unit_class=cls, statistic=stat,
                              mean=v.mean(), sd=v.std(ddof=1), median=np.median(v), min=v.min(), max=v.max(),
                              n_networks=len(v), n_sig_pos=int(((d[stat.replace('raw_r', 'partial_r')] > 0) &
                                                                (d['partial_r_p'] < 0.05)).sum()) if stat == 'partial_r' else np.nan,
                              n_sig_neg=int(((d['partial_r'] < 0) & (d['partial_r_p'] < 0.05)).sum()) if stat == 'partial_r' else np.nan,
                              test='Wilcoxon signed-rank vs 0 across networks', p=w.pvalue if w else np.nan))
    for ts in ('dT_le_0', 'all_dT'):
        for cls in ('E', 'I'):
            for stat in ('partial_r', 'bias_coef', 'bias_coef_celladj'):
                b = per.query("group=='bilateral' and condition=='intact' and trial_set==@ts and unit_class==@cls")[stat].dropna()
                n = per.query("group=='no_commissure' and condition=='intact' and trial_set==@ts and unit_class==@cls")[stat].dropna()
                srows.append(dict(group='bilateral_vs_no_commissure', condition='intact', trial_set=ts, unit_class=cls,
                                  statistic=stat, mean=b.mean() - n.mean(), n_networks=f'{len(b)}+{len(n)}',
                                  test='Mann-Whitney U (two-sided); mean = difference of group means',
                                  p=stats.mannwhitneyu(b, n).pvalue))
    for grp in ('bilateral', 'no_commissure'):
        for cond in ('no_shared_input_noise', 'commissure_cut'):
            a0 = per.query("group==@grp and condition=='intact' and trial_set=='dT_le_0' and unit_class=='E'")
            a1 = per.query("group==@grp and condition==@cond and trial_set=='dT_le_0' and unit_class=='E'")
            if len(a1) == 0:
                continue
            dlt = a1.partial_r.to_numpy() - a0.partial_r.to_numpy()
            srows.append(dict(group=grp, condition=f'{cond}_minus_intact', trial_set='dT_le_0', unit_class='E',
                              statistic='partial_r_change', mean=dlt.mean(), sd=dlt.std(ddof=1), n_networks=len(dlt),
                              test='paired Wilcoxon signed-rank', p=stats.wilcoxon(dlt).pvalue))
    for r in mr.itertuples():
        srows.append(dict(group=f'monkey_{r.session}', condition='recorded', trial_set='all', unit_class='all',
                          statistic='partial_r', mean=r.value, n_networks=np.nan, test=r.test, p=r.p))
    for r in mb.itertuples():
        srows.append(dict(group=f'monkey_{r.session}', condition='recorded', trial_set='choice', unit_class='all',
                          statistic='bias_coef', mean=r.value, n_networks=np.nan, test=r.test, p=r.p))
        srows.append(dict(group=f'monkey_{r.session}', condition='recorded', trial_set='choice', unit_class='all',
                          statistic='bias_z', mean=r.z, n_networks=np.nan, test=r.test, p=r.p))
    S = pd.DataFrame(srows)
    S.to_csv(OUT / 'stats.csv', index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 20, 'display.max_rows', 500):
        print(S[['group', 'condition', 'trial_set', 'unit_class', 'statistic', 'mean', 'sd', 'n_sig_pos',
                 'n_sig_neg', 'p']].to_string())


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--summarize':
        summarize(pd.read_csv(OUT / 'P5_model_per_network.csv'), pd.read_csv(OUT / 'P5_model_tercile_per_network.csv'))
    else:
        main()
