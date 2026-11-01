#!/usr/bin/env python3
"""
Build B5 (Aim 1 panels): is the RT-predictive pre-target SC state location-specific?

Re-runs the logic of scripts/cp_p2c_rt_specificity.py (which printed but did not save its results) and the z_target
regressor of scripts/cp_p2_conflict.py (c), saving everything. Per unit: pre-target rate (-300..0 ms re target onset),
z-scored within block type (high-probability location = export loc 1 or loc 3), unit-stable verified trials. Per trial:
mean z over units grouped by where their preferred contralateral location P (g1_geometry) lies:
  z_prior   P = the block's high-probability location
  z_away    P = the location opposite the high-probability location
  z_other   P is not a probability location (export loc 2 / 4)
  z_all     all units
  z_target  P = the target location (added, from cp_p2_conflict.py)
Trials split by target at the high-probability location ('high') vs elsewhere ('low'). OLS log RT ~ session + target
location + reward + z (single regressor), and the joint model ~ ... + z_prior + z_away (partial coefficients), plain OLS
as in the source script; cluster-robust (block run) p values added. Coefficient x 100 = % RT per SD (100 x d log RT).
Run: OMP_NUM_THREADS=1 nice -n 19 python scripts/aim1_panels/build_B5.py   (~1-2 min)
"""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.sessions import ALL_SESSIONS  # noqa: E402
from src.intercollicular import session_table, counts  # noqa: E402
from src.competition import trial_table  # noqa: E402

OUT = REPO / 'results' / 'aim1_panels' / 'B5_rt_state_specificity'
G1 = REPO / 'results' / 'intercollicular' / 'g1_geometry' / 'rows.npy'
OPP = {1: 3, 3: 1, 2: 4, 4: 2}
# (cond, regressor) -> (row label, role under local competition)
ROLE = {('high', 'z_prior'): ('RF at target (= expected loc.)', 'target'),
        ('high', 'z_away'): ('RF at opposite (non-target)', 'competitor'),
        ('high', 'z_other'): ('RF at non-probability loc. (non-target)', 'competitor'),
        ('high', 'z_all'): ('all units', 'mixed'),
        ('low', 'z_target'): ('RF at target (unexpected loc.)', 'target'),
        ('low', 'z_prior'): ('RF at expected loc. (competitor)', 'competitor'),
        ('low', 'z_away'): ('RF at opposite of expected (target there on 1/3 of trials)', 'mixed'),
        ('low', 'z_other'): ('RF at non-probability loc. (target there on 1/3 of trials)', 'mixed'),
        ('low', 'z_away_nontarget'): ('RF at opposite of expected, target elsewhere', 'competitor'),
        ('low', 'z_other_nontarget'): ('RF at non-probability loc., target elsewhere', 'competitor'),
        ('low', 'z_all'): ('all units', 'mixed')}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / 'scripts' / 'aim1_panels' / 'style.py', OUT / 'style.py')
    g1 = {(r['session'], r['id']): r for r in np.load(G1, allow_pickle=True)}
    rows, nunits = [], []
    for s in ALL_SESSIONS:
        if s == '08_15':
            continue
        t = session_table(s)
        tt = trial_table(s)
        n = len(t['loc'])
        hp = np.full(n, -1)
        rt = np.full(n, np.nan)
        rew = np.full(n, -1)
        blk = np.full(n, -1.0)
        m = tt['export_idx'] >= 0
        hp[tt['export_idx'][m]] = tt['high_p_loc'][m]
        rt[tt['export_idx'][m]] = tt['rt'][m] * 1000
        rew[tt['export_idx'][m]] = tt['reward_dur'][m] > 200
        blk[tt['export_idx'][m]] = tt['block'][m]
        run = np.cumsum(np.r_[0, np.diff(blk) != 0])
        on = t['ev']['targetOn']
        Z = {}
        for j, i in enumerate(t['ids']):
            P = g1[(s, i)]['P']
            stab = t['stable'][:, j]
            r = counts(t['spikes'][i], on - 0.3, on) / 0.3
            z = np.full(n, np.nan)
            for b in (1, 3):
                sel = stab & (hp == b)
                if sel.sum() > 10 and np.nanstd(r[sel]) > 0:
                    z[sel] = (r[sel] - np.nanmean(r[sel])) / np.nanstd(r[sel])
            Z[(i, P, t['type'][j])] = z
            nunits.append(dict(session=s, unit_id=int(i), unit_type=t['type'][j], P_loc=int(P)))
        loc = t['loc']
        for k in range(n):
            if hp[k] < 0 or np.isnan(rt[k]):
                continue

            def grp(f):
                v = [z[k] for (i, P, ty), z in Z.items() if f(P) and not np.isnan(z[k])]
                return (np.mean(v) if v else np.nan), len(v)
            zp, npr = grp(lambda P: P == hp[k])
            za, naw = grp(lambda P: P == OPP[hp[k]])
            zo, nob = grp(lambda P: P not in (1, 3))
            zl, nal = grp(lambda P: True)
            zt, nta = grp(lambda P: P == loc[k])
            zan, _ = grp(lambda P: P == OPP[hp[k]] and P != loc[k])
            zon, _ = grp(lambda P: P not in (1, 3) and P != loc[k])
            rows.append(dict(session=s, trial_export_idx=k, block_run=f'{s}_{run[k]}', rt_ms=rt[k], log_rt=np.log(rt[k]),
                             target_loc=int(loc[k]), high_p_loc=int(hp[k]), reward_high=int(rew[k]),
                             cond='high' if loc[k] == hp[k] else 'low',
                             z_prior=zp, z_away=za, z_other=zo, z_all=zl, z_target=zt,
                             z_away_nontarget=zan, z_other_nontarget=zon,
                             n_units_prior=npr, n_units_away=naw, n_units_other=nob, n_units_all=nal, n_units_target=nta))
        print(s, 'done', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'B5_trials.csv', index=False, float_format='%.5g')
    pd.DataFrame(nunits).to_csv(OUT / 'B5_units.csv', index=False)

    coefs = []
    for cond in ('high', 'low'):
        d = df[df.cond == cond]
        for v in ('z_prior', 'z_away', 'z_other', 'z_all', 'z_target', 'z_away_nontarget', 'z_other_nontarget'):
            if (cond, v) not in ROLE:
                continue
            dd = d.dropna(subset=[v])
            f = smf.ols(f'log_rt ~ C(session) + C(target_loc) + reward_high + {v}', dd).fit()
            fc = smf.ols(f'log_rt ~ C(session) + C(target_loc) + reward_high + {v}', dd).fit(
                cov_type='cluster', cov_kwds={'groups': dd['block_run']})
            ci = f.conf_int().loc[v] * 100
            coefs.append(dict(model='single', target_cond=cond, regressor=v, label=ROLE[(cond, v)][0],
                              competition_role=ROLE[(cond, v)][1], coef_pct_per_sd=100 * f.params[v], ci95_lo=ci[0],
                              ci95_hi=ci[1], p_ols=f.pvalues[v], p_cluster_block=fc.pvalues[v], n_trials=len(dd),
                              sessions=';'.join(dd.session.unique())))
        # joint models (partial coefficients). The source script's joint model (z_prior + z_away) has no trials by
        # construction: P is always a contralateral location (export loc 1 or 2), so z_prior exists only in
        # loc-1-high blocks and z_away only in loc-3-high blocks. z_prior + z_other is the estimable joint model.
        for pair in (('z_prior', 'z_away'), ('z_prior', 'z_other')):
            dd = d.dropna(subset=list(pair))
            if len(dd) < 50:
                for v in pair:
                    coefs.append(dict(model=f'joint ({pair[0]} + {pair[1]})', target_cond=cond, regressor=v,
                                      label=ROLE[(cond, v)][0], competition_role=ROLE[(cond, v)][1], n_trials=len(dd),
                                      coef_pct_per_sd=np.nan, ci95_lo=np.nan, ci95_hi=np.nan, p_ols=np.nan,
                                      p_cluster_block=np.nan, sessions='not estimable (no trial has both regressors)'))
                continue
            form = f'log_rt ~ C(session) + C(target_loc) + reward_high + {pair[0]} + {pair[1]}'
            f = smf.ols(form, dd).fit()
            fc = smf.ols(form, dd).fit(cov_type='cluster', cov_kwds={'groups': dd['block_run']})
            for v in pair:
                ci = f.conf_int().loc[v] * 100
                coefs.append(dict(model=f'joint ({pair[0]} + {pair[1]})', target_cond=cond, regressor=v,
                                  label=ROLE[(cond, v)][0], competition_role=ROLE[(cond, v)][1],
                                  coef_pct_per_sd=100 * f.params[v], ci95_lo=ci[0], ci95_hi=ci[1], p_ols=f.pvalues[v],
                                  p_cluster_block=fc.pvalues[v], n_trials=len(dd), sessions=';'.join(dd.session.unique())))
    c = pd.DataFrame(coefs)
    c.to_csv(OUT / 'B5_coefs.csv', index=False, float_format='%.5g')
    commit = subprocess.run(['git', '-C', str(REPO), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
    st = pd.DataFrame(dict(statistic=[f'RT % per SD, {r.regressor}, target {r.target_cond} ({r.model})' for r in c.itertuples()],
                           value=c.coef_pct_per_sd, ci95_lo=c.ci95_lo, ci95_hi=c.ci95_hi, p=c.p_ols,
                           p_cluster_block=c.p_cluster_block, n=c.n_trials))
    st['test'] = 'OLS log RT ~ session + target location + reward + z (x100); p plain OLS (as cp_p2c), cluster-robust by block run also given'
    st['window'] = 'pre-target -300..0 ms re target onset (z within block type per unit)'
    st['unit_class'] = 'E and I pooled'
    st['session_filter'] = 'diagonal-target sessions with units (Newton 08_15 excluded); verified successful trials'
    st['source'] = f'recomputed (scripts/aim1_panels/build_B5.py, logic of scripts/cp_p2c_rt_specificity.py), git {commit}'
    st.to_csv(OUT / 'stats.csv', index=False, float_format='%.5g')
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        print(c.drop(columns=['sessions']).round(4).to_string())
        print(df.groupby('cond')[['z_prior', 'z_away', 'z_other', 'z_target']].count())


if __name__ == '__main__':
    main()
