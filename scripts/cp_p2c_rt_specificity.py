"""P2c: is the RT-predictive pre-target SC state location-specific? (units grouped by RF relative to the high-probability location)."""
import sys; sys.path.insert(0, r'D:\Projects\4factors-rnn-analysis')
import numpy as np, pandas as pd, statsmodels.formula.api as smf
from src.sessions import ALL_SESSIONS
from src.intercollicular import session_table, counts
from src.competition import trial_table
g1 = {(r['session'], r['id']): r for r in np.load(r'D:\Projects\4factors-rnn-analysis\results\intercollicular\g1_geometry\rows.npy', allow_pickle=True)}
OPP = {1: 3, 3: 1, 2: 4, 4: 2}
rows = []
for s in ALL_SESSIONS:
    if s == '08_15': continue
    t = session_table(s); tt = trial_table(s); n = len(t['loc'])
    hp = np.full(n, -1); rt = np.full(n, np.nan); rew = np.full(n, -1)
    m = tt['export_idx'] >= 0
    hp[tt['export_idx'][m]] = tt['high_p_loc'][m]; rt[tt['export_idx'][m]] = tt['rt'][m] * 1000; rew[tt['export_idx'][m]] = tt['reward_dur'][m] > 200
    on = t['ev']['targetOn']; Z = {}
    for j, i in enumerate(t['ids']):
        P = g1[(s, i)]['P']; stab = t['stable'][:, j]
        r = counts(t['spikes'][i], on - 0.3, on) / 0.3
        z = np.full(n, np.nan)
        for b in (1, 3):
            sel = stab & (hp == b)
            if sel.sum() > 10 and np.nanstd(r[sel]) > 0: z[sel] = (r[sel] - np.nanmean(r[sel])) / np.nanstd(r[sel])
        Z[(i, P, t['type'][j])] = z
    for k in range(n):
        if hp[k] < 0 or np.isnan(rt[k]): continue
        grp = lambda f: np.nanmean([z[k] for (i, P, ty), z in Z.items() if f(P, ty)] or [np.nan])
        rows.append(dict(session=s, log_rt=np.log(rt[k]), loc=int(t['loc'][k]), reward=rew[k],
                         cond='high' if t['loc'][k] == hp[k] else 'low',
                         z_prior=grp(lambda P, ty: P == hp[k]), z_away=grp(lambda P, ty: P == OPP[hp[k]]),
                         z_other=grp(lambda P, ty: P not in (1, 3)), z_all=grp(lambda P, ty: True)))
df = pd.DataFrame(rows)
for cond in ('high', 'low'):
    d = df[df.cond == cond]
    for v in ('z_prior', 'z_away', 'z_other', 'z_all'):
        dd = d.dropna(subset=[v])
        if len(dd) < 50: print(cond, v, 'n', len(dd)); continue
        f = smf.ols(f'log_rt ~ C(session) + C(loc) + reward + {v}', dd).fit()
        print(f'{cond:4s} {v:8s} n={len(dd):4d} coef {100*f.params[v]:+.2f}%/SD p={f.pvalues[v]:.2g}')
    dd = d.dropna(subset=['z_prior', 'z_away'])
    print(cond, 'joint n', len(dd), 'sessions', dd.session.unique())
    if len(dd) < 50: continue
    f = smf.ols('log_rt ~ C(session) + C(loc) + reward + z_prior + z_away', dd).fit()
    print(f'{cond} joint: prior {100*f.params["z_prior"]:+.2f} (p={f.pvalues["z_prior"]:.2g}), away {100*f.params["z_away"]:+.2f} (p={f.pvalues["z_away"]:.2g}), n={len(dd)}')
