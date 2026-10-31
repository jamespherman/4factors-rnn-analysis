#!/usr/bin/env python3
"""
Per-trial value belief for the srsClassic conflict task (plans/2026-10-31_hub_resolver_models_plan.md).

Fits, per session, models of the monkey's choices on two-target trials:
  M_none:   P(right) = lapse/2 + (1 - lapse) * sigmoid(b0 + s * sal_right)
  M_oracle: + beta * (true block-mean reward of the right identity - left identity) / 100 ms
  M_delta:  + beta * (V_right - V_left) / 100 ms, where V_T1, V_T2 are delta-rule estimates updated from the reward
            received for the chosen/shown identity on every rewarded trial (single and choice), learning rate alpha,
            initialised at the session's first-block mean
  M_mix:    P(right) = w * sigmoid(beta * dV) + (1 - w) * sigmoid(b0 + s * sal_right): a value-driven policy on a
            fraction w of trials and a side/salience-biased policy otherwise (one route to weak reward use)
Parameters are bounded (beta <= 20 per 100 ms, alpha in [0.01, 1]) to avoid degenerate deterministic fits.
sal_right = +1 if the right target is high salience, else -1. Maximum likelihood; AIC. Secondary check: single-target RT
and peak velocity vs the shown identity's belief value. Writes per-trial beliefs (value before the trial) for the
trial-matched simulations.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

SRC = Path(r'D:/Google Drive/HermanLab/data/Newton_SC_2026/data/export')
OUT = Path(__file__).parent.parent / 'results' / 'hub_models' / 'belief'
SESSIONS = {'08-19': 'Newton_08192026_srsClassic_trials.csv', '08-20': 'Newton_08202026_srsClassic_trials.csv'}


def delta_values(d: pd.DataFrame, alpha: float) -> np.ndarray:
    """[n, 2] belief (ms) for T1, T2 before each trial."""
    v = np.full(2, np.nanmean(np.r_[d['t1_reward_ms'], d['t2_reward_ms']]))
    out = np.zeros((len(d), 2))
    for k, r in enumerate(d.itertuples()):
        out[k] = v
        if r.rewarded == 1 and not np.isnan(r.chosen_target_id):
            i = int(r.chosen_target_id) - 1
            v[i] += alpha * (r.reward_duration_ms - v[i])
    return out


def block_means(d: pd.DataFrame) -> np.ndarray:
    """True per-block mean reward of T1, T2 (mean of the per-trial draws within the block)."""
    out = np.zeros((len(d), 2))
    for b, g in d.groupby('conflict_block'):
        out[g.index, 0] = g['t1_reward_ms'].mean()
        out[g.index, 1] = g['t2_reward_ms'].mean()
    return out


def design(d, vals):
    """For choice trials: value difference right-left (per 100 ms), salience sign, chose-right."""
    right_is_t1 = d['t1_side'] == 1
    v_right = np.where(right_is_t1, vals[:, 0], vals[:, 1])
    v_left = np.where(right_is_t1, vals[:, 1], vals[:, 0])
    sal_right = np.where(right_is_t1, d['t1_salience_contrast'], d['t2_salience_contrast'])
    return (v_right - v_left) / 100, np.where(sal_right == 180, 1.0, -1.0), (d['chosen_side'] == 1).astype(float)


def unpack(params, kind):
    sig = lambda x: 1 / (1 + np.exp(-x))
    out = {'b0': params[0], 's': params[1]}
    if kind in ('none', 'oracle', 'delta'):
        out['lapse'] = 0.5 * sig(params[2])
    if kind in ('oracle', 'delta', 'mix'):
        out['beta'] = 20 * sig(params[3])
    if kind in ('delta', 'mix'):
        out['alpha'] = 0.01 + 0.99 * sig(params[4])
    if kind == 'mix':
        out['w'] = sig(params[2])
    return out


def predict(params, d, kind):
    q = unpack(params, kind)
    if kind == 'none':
        vals = np.zeros((len(d), 2))
    elif kind == 'oracle':
        vals = block_means(d)
    else:
        vals = delta_values(d, q['alpha'])
    dv, sal, y = design(d, vals)
    bias = 1 / (1 + np.exp(-(q['b0'] + q['s'] * sal)))
    if kind == 'mix':
        p = q['w'] / (1 + np.exp(-q['beta'] * dv)) + (1 - q['w']) * bias
    else:
        p = q['lapse'] / 2 + (1 - q['lapse']) / (1 + np.exp(-(q['b0'] + q['s'] * sal + q.get('beta', 0) * dv)))
    return p, y, vals


def nll(params, d, choice, kind):
    p, y, _ = predict(params, d, kind)
    p = np.clip(p[choice], 1e-9, 1 - 1e-9)
    return -np.sum(y[choice] * np.log(p) + (1 - y[choice]) * np.log(1 - p))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    L = ['# Value-belief fits (srsClassic, two-target choice trials; RT >= 80 ms)\n']
    for ses, fn in SESSIONS.items():
        d = pd.read_csv(SRC / fn).reset_index(drop=True)
        choice = ((d['n_targets'] == 2) & d['outcome'].isin(['CHOSE_HIGH_SAL', 'CHOSE_LOW_SAL'])
                  & (d['rt_s'] >= 0.08) & d['chosen_side'].isin([1, 2])).to_numpy()
        fits = {}
        for kind, x0 in [('none', [0, 0, -4]), ('oracle', [0, 0, -4, -2]), ('delta', [0, 0, -4, -2, -1]),
                         ('mix', [0, 0, 0, 0, -1])]:
            best = None
            for jitter in range(8):
                start = np.array(x0, float) + np.random.default_rng(jitter).normal(0, 0.5, len(x0)) * (jitter > 0)
                r = minimize(nll, start, args=(d, choice, kind), method='Nelder-Mead',
                             options={'maxiter': 4000, 'xatol': 1e-4, 'fatol': 1e-6})
                if best is None or r.fun < best.fun:
                    best = r
            fits[kind] = best
        L.append(f'## {ses}: {choice.sum()} choice trials')
        for kind, r in fits.items():
            k = len(r.x)
            par = unpack(r.x, kind)
            L.append(f'- {kind:6s}: NLL {r.fun:.1f}, AIC {2 * r.fun + 2 * k:.1f}; ' + ', '.join(f'{a} {b:+.3f}' for a, b in par.items()))
        best_kind = min(('delta', 'mix'), key=lambda k: fits[k].fun)
        alpha = unpack(fits[best_kind].x, best_kind)['alpha']
        L.append(f'- beliefs from {best_kind} fit (alpha {alpha:.3f})')
        vals = delta_values(d, alpha)
        bm = block_means(d)
        d['belief_t1_ms'], d['belief_t2_ms'] = vals[:, 0], vals[:, 1]
        d['block_mean_t1_ms'], d['block_mean_t2_ms'] = bm[:, 0], bm[:, 1]
        # how good is the belief: fraction of choice trials where belief ranks identities correctly
        rich = np.where(bm[:, 0] > bm[:, 1], 1, 2)
        bel = np.where(vals[:, 0] > vals[:, 1], 1, 2)
        L.append(f'- delta belief ranks the rich identity correctly on {np.mean(bel[choice] == rich[choice]):.0%} of choice trials')
        # single-target vigor/RT vs belief of the shown identity
        st = ((d['n_targets'] == 1) & d['outcome'].isin(['SINGLE_T1_CORRECT', 'SINGLE_T2_CORRECT'])
              & (d['rt_s'] >= 0.08)).to_numpy()
        shown = np.where(d['outcome'] == 'SINGLE_T1_CORRECT', 0, 1)
        v_shown = vals[np.arange(len(d)), shown]
        for col in ('rt_s', 'sac_peak_velocity'):
            r, p = spearmanr(v_shown[st], d[col].to_numpy()[st], nan_policy='omit')
            L.append(f'- single-target {col} vs belief of shown identity: Spearman {r:+.3f} (p={p:.2g}, n={st.sum()})')
        d.to_csv(OUT / f'{ses}_trials_with_belief.csv', index=False)
        L.append('')
    text = '\n'.join(L)
    (OUT / 'summary.md').write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
