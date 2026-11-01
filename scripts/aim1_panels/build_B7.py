#!/usr/bin/env python3
"""Build the B7 data folder: reward-difference and choice information in the left-minus-right SC signal (i), and
competitor-reward normalisation per unit (ii), Newton bilateral srsClassic 08-20 (08-19 in the CSVs, descriptive).

Sources (read-only; MATLAB tables exported by scripts/aim1_panels/export_bilateral_tables.m):
  _bilateral_source/competition_choice_tc_<session>.csv    <- Newton_SC_2026 code/run_competition_choice.m
  _bilateral_source/competition_normalisation_units.csv     <- code/run_competition_normalisation.m (per-unit coefs)
  _bilateral_source/competition_normalisation_sims.csv      <- same (200 null / planted simulations of the unit mean)
  results/hub_models/sig_v2/<net>/draws.csv                 <- scripts/srs_stage0.py signatures S1, S2 on the trained
                                                               v2 networks (10 simulated sessions per network)
Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 python scripts/aim1_panels/build_B7.py
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, t as tdist

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'results' / 'aim1_panels' / '_bilateral_source'
OUT = ROOT / 'results' / 'aim1_panels' / 'B7_lr_signal_normalisation'
OUT.mkdir(parents=True, exist_ok=True)
shutil.copy(ROOT / 'scripts' / 'aim1_panels' / 'style.py', OUT / 'style.py')
N_WIN = 11                          # placebo x2 + nine 50-ms post-target windows (pre-saccade is the late reference)
ALPHA = 0.05
stats = []


def stat(name, value, n, test, window, unit_class, session, ci_lo=np.nan, ci_hi=np.nan, p=np.nan, note=''):
    stats.append(dict(statistic=name, value=value, ci_lo=ci_lo, ci_hi=ci_hi, p=p, n=n, test=test, window=window,
                      unit_class=unit_class, session=session, note=note))


# ---------------- (i) time course ----------------
rows, nulls = [], []
for key, ses in (('Newton_08202026', '08-20'), ('Newton_08192026', '08-19')):
    tc = pd.read_csv(SRC / f'competition_choice_tc_{key}.csv')
    for _, r in tc.iterrows():
        w = r['window']
        if w.startswith('placebo'):
            a, b = [int(x) for x in w.replace('placebo ', '').split('..')]
            ref, kind = 'target', 'placebo (before target)'
        elif w.startswith('target'):
            a, b = [int(x) for x in w.replace('target ', '').replace('+', '').split('..')]
            ref, kind = 'target', 'post-target'
        else:
            a, b, ref, kind = -100, 0, 'saccade', 'pre-saccade reference (not in the 11-window family)'
        n = int(r['n'])
        rows.append(dict(session=ses, window=w, window_kind=kind, aligned_to=ref, t_start_ms=a, t_end_ms=b,
                         t_center_ms=(a + b) / 2, n_trials=n, choice_LR=r['LR'], choice_p=r['p'],
                         choice_z_signed=np.sign(r['added_CP'] - 0.5) * np.sqrt(max(r['LR'], 0)),
                         added_CP=r['added_CP'], reward_t=r['reward_t'],
                         reward_p_two_sided=2 * tdist.sf(abs(r['reward_t']), n - 5),
                         choice_sig_bonferroni=bool(r['p'] < ALPHA / N_WIN) if ref == 'target' else np.nan))
    n = int(tc['n'].iloc[0])
    for lvl, a_ in (('uncorrected 95% (two-sided)', ALPHA), (f'Bonferroni over {N_WIN} windows (two-sided 0.05)',
                                                               ALPHA / N_WIN)):
        nulls.append(dict(session=ses, level=lvl, alpha_two_sided=a_, n_trials=n,
                          choice_z_crit=norm.isf(a_ / 2),
                          reward_t_crit=tdist.isf(a_ / 2, n - 5),
                          note='choice: z = sign(added CP - 0.5) x sqrt(LR chi2, 1 df); reward: OLS t, df = n - 5'))
T = pd.DataFrame(rows)
T.to_csv(OUT / 'B7_i_timecourse.csv', index=False)
NB = pd.DataFrame(nulls)
NB.to_csv(OUT / 'B7_i_null_bands.csv', index=False)
t20 = T[T.session == '08-20']
crit = NB[(NB.session == '08-20') & NB.level.str.startswith('Bonf')].iloc[0]
for _, r in t20.iterrows():
    stat(f'added CP, {r.window}', r.added_CP, r.n_trials, 'AUC of window L-R signal residualised on M0; LR test p',
         r.window, 'all units, L-R population', '08-20', p=r.choice_p)
    stat(f'reward-difference t, {r.window}', r.reward_t, r.n_trials, 'OLS W ~ reward diff + salience diff + choice + '
         'pre-target bias; t of reward diff', r.window, 'all units, L-R population', '08-20', p=r.reward_p_two_sided)
post = t20[t20.aligned_to == 'target']
stat('max |reward-difference t| over the 11 windows', post.reward_t.abs().max(), len(post), 'max |t|', '11 windows',
     'all units, L-R population', '08-20', note=f'Bonferroni critical |t| = {crit.reward_t_crit:.2f}; write-up: '
     '"no reward difference in any window" (peer note |t| < 1.9)')
stat('windows with choice LR p < 0.05/11', int((post.choice_p < ALPHA / N_WIN).sum()), len(post), 'LR test, '
     'Bonferroni', '11 windows', 'all units, L-R population', '08-20', note='+100..150 (p 0.0037) and +350..400 '
     '(p 0.0018)')

# trained networks' S1 (reward-difference t in the L-R signal; delay 100-450 ms and pre-saccadic -100..0 ms)
net_rows, draw_rows = [], []
for f in sorted((ROOT / 'results' / 'hub_models' / 'sig_v2').glob('*/draws.csv')):
    d = pd.read_csv(f)
    net = d.arch.iloc[0]
    arch = net.split('_')[0]
    role = {'A': 'resolver', 'AE': 'resolver', 'B': 'priority hub', 'C': 'outcome hub'}[arch]
    for _, r in d.iterrows():
        draw_rows.append(dict(network=net, architecture=arch, role=role, draw=int(r.draw), S1_delay_t=r.S1_delay,
                              S1_presac_t=r.S1_presac, S2_mean_comp_coef=r.S2_mean, S2_t=r.S2_t))
    net_rows.append(dict(network=net, architecture=arch, role=role, n_draws=len(d),
                         S1_delay_t_mean=d.S1_delay.mean(), S1_delay_t_sd=d.S1_delay.std(),
                         S1_presac_t_mean=d.S1_presac.mean(), S1_presac_t_sd=d.S1_presac.std(),
                         S2_mean_comp_coef_mean=d.S2_mean.mean(), S2_mean_comp_coef_sd=d.S2_mean.std(),
                         S2_t_mean=d.S2_t.mean()))
NET = pd.DataFrame(net_rows)
NET.to_csv(OUT / 'B7_model_S1_S2_per_network.csv', index=False)
pd.DataFrame(draw_rows).to_csv(OUT / 'B7_model_S1_S2_draws.csv', index=False)
for role in ('resolver', 'priority hub', 'outcome hub'):
    g = NET[NET.role == role]
    for col, lab in (('S1_delay_t_mean', 'S1 reward-difference t, delay 100-450 ms'),
                     ('S1_presac_t_mean', 'S1 reward-difference t, pre-saccadic -100..0 ms'),
                     ('S2_mean_comp_coef_mean', 'S2 mean competitor-reward coefficient (log rate per 100 ms), '
                                                'delay 100-450 ms')):
        stat(f'trained {role} networks: {lab} (mean over networks)', g[col].mean(), len(g),
             'per network mean over 10 simulated sessions at recorded unit/trial counts (sig_v2)', lab.split(', ')[-1],
             'model units sampled like recorded', 'model (08-20 trial sequence)', g[col].min(), g[col].max(),
             note='ci_lo/ci_hi = range over networks')

# ---------------- (ii) per-unit competitor-reward coefficients ----------------
U = pd.read_csv(SRC / 'competition_normalisation_units.csv')
U['session'] = U.session.map({'Newton_08202026': '08-20', 'Newton_08192026': '08-19'})
U = U.rename(columns={'unit': 'unit_row', 'comp_a': 'comp_coef_model_a', 'comp_b': 'comp_coef_model_b'})
U.to_csv(OUT / 'B7_ii_unit_coefficients.csv', index=False)
S = pd.read_csv(SRC / 'competition_normalisation_sims.csv')
S['session'] = S.session.map({'Newton_08202026': '08-20', 'Newton_08192026': '08-19'})
S.to_csv(OUT / 'B7_ii_sims_of_unit_mean.csv', index=False)
summ = []
for (ses, ep, model), g in S.groupby(['session', 'epoch', 'model']):
    real = U[(U.session == ses) & (U.epoch == ep)][f'comp_coef_model_{model}'].mean()
    nU = int(((U.session == ses) & (U.epoch == ep)).sum())
    null = g[g.gamma == 0].mean_comp_coef
    lo = np.percentile(null, 2.5)
    row = dict(session=ses, epoch=ep, model=model, n_units=nU, real_mean=real, null_mean=null.mean(),
               null_q025=lo, null_q975=np.percentile(null, 97.5))
    for gm in (0.1, 0.2):
        pl = g[g.gamma == gm].mean_comp_coef
        row[f'planted_g{gm}_mean'] = pl.mean()
        row[f'planted_g{gm}_q025'] = np.percentile(pl, 2.5)
        row[f'planted_g{gm}_q975'] = np.percentile(pl, 97.5)
        row[f'power_g{gm}'] = float(np.mean(pl < lo))
    summ.append(row)
SM = pd.DataFrame(summ)
SM.to_csv(OUT / 'B7_ii_population_mean_summary.csv', index=False)
for _, r in SM[SM.session == '08-20'].iterrows():
    stat(f'mean competitor-reward coefficient, model ({r.model}), {r.epoch}', r.real_mean, r.n_units,
         'Poisson GLM per unit; mean over units vs 200 sims of pure choice coding (null 2.5-97.5%)',
         {'visual': '+40..+300 ms re target', 'delay': '+300..+467 ms re target', 'presacc': '-100..0 ms re saccade'}[
             r.epoch], 'all units with >= 30 valid trials and mean count >= 0.5 (contra frame)', '08-20',
         r.null_q025, r.null_q975, note=f'null mean {r.null_mean:+.3f}; power gamma 0.1 = {r["power_g0.1"]:.3f}, gamma 0.2 = '
         f'{r["power_g0.2"]:.3f}; model a = without choice, b = + chose-contra')
pd.DataFrame(stats).to_csv(OUT / 'stats.csv', index=False)
print(T[T.session == '08-20'][['window', 'added_CP', 'choice_p', 'choice_z_signed', 'reward_t']].round(3).to_string())
print(NB.round(3).to_string())
print(SM.round(4).to_string())
print(NET.groupby('role')[['S1_delay_t_mean', 'S1_presac_t_mean', 'S2_mean_comp_coef_mean']].agg(['mean', 'min', 'max'])
      .round(3).to_string())
