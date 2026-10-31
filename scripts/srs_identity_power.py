#!/usr/bin/env python3
"""
Power of the S8 identity test (recorded SC vs trained resolver) as a function of recording yield, for planning the
proposed bilateral recordings (results/AIM1_FINDINGS_WRITEUP.md, C3).

Per simulated data set: n_units contralateral units, n_blocks blocks (alternating rich identity), n_per single-target
trials per identity per block. Unit rates ~ lognormal (median 4 sp/s, SD 0.8 in log) in a 350 ms delay window;
trial-level lognormal gain (SD 0.4) for recorded-like overdispersion. Each unit's true identity effect beyond value
(log T1/T2, block-invariant) is drawn from the pooled trained-resolver distribution (A and AE, delay window,
results/hub_models/identity_delta_true_v2.csv), optionally scaled (weaker resolvers) or with only a fraction of units
coding (sparser resolvers). Statistic: mean over units of |mean over blocks of log((m1 + 0.5)/(m2 + 0.5))|, against its
distribution with no identity effect (same rates and trial counts; equivalent to the identity-shuffle null).
Output: results/hub_models/identity_power/power.csv and summary.md.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
OUT = ROOT / 'results' / 'hub_models' / 'identity_power'
WIN, RATE_MED, RATE_SD, OVERDISP = 0.35, 4.0, 0.8, 0.4
N_PER = 19            # completed single-target trials per identity per block (08-20: 301 trials / 8 blocks / 2)
N_SIM = 400


def stat(rng, n_units, n_blocks, delta):
    """delta [n_units] true log effects -> statistic."""
    rate = np.exp(np.log(RATE_MED) + RATE_SD * rng.standard_normal(n_units))[:, None, None]
    d = []
    for ident, sgn in ((0, 0.5), (1, -0.5)):
        g = np.exp(OVERDISP * rng.standard_normal((n_units, n_blocks, N_PER)) - OVERDISP ** 2 / 2)
        lam = rate * np.exp(sgn * delta[:, None, None]) * g * WIN
        d.append(rng.poisson(lam).mean(-1))
    est = np.log((d[0] + 0.5) / (d[1] + 0.5)).mean(1)
    return np.abs(est).mean()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = pd.read_csv(ROOT / 'results' / 'hub_models' / 'identity_delta_true_v2.csv')
    pool = t[(t['window'] == 'delay') & t['model'].str.match(r'^(A|AE)_s')]['delta_true'].to_numpy()
    pool = np.clip(pool[np.isfinite(pool)], -3, 3)        # near-silent model units can give extreme log ratios
    rng = np.random.default_rng(0)
    rows = []
    for n_units in (10, 20, 30, 50, 80, 120):
        for n_blocks in (2, 4, 8, 12):
            null = np.array([stat(rng, n_units, n_blocks, np.zeros(n_units)) for _ in range(N_SIM)])
            crit = np.quantile(null, 0.95)
            for label, scale, frac in (('trained resolver', 1.0, 1.0), ('half strength', 0.5, 1.0),
                                       ('quarter of units code', 1.0, 0.25), ('half strength, half of units', 0.5, 0.5)):
                hits = 0
                for _ in range(N_SIM):
                    dl = rng.choice(pool, n_units) * scale * (rng.random(n_units) < frac)
                    hits += stat(rng, n_units, n_blocks, dl) > crit
                rows.append(dict(n_units=n_units, n_blocks=n_blocks, resolver=label, power=hits / N_SIM))
        print(n_units, 'done', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'power.csv', index=False)
    L = ['# Power of the S8 identity test vs recording yield (alpha 0.05; delay window)\n',
         f'Resolver effects drawn from trained A/AE networks (median |delta| {np.median(np.abs(pool)):.2f}); '
         f'{N_PER} trials per identity per block; rates lognormal median {RATE_MED} sp/s; overdispersion {OVERDISP}.\n']
    for label, g in df.groupby('resolver', sort=False):
        L.append(f'## {label}')
        p = g.pivot(index='n_units', columns='n_blocks', values='power')
        L.append('| units \\ blocks | ' + ' | '.join(str(c) for c in p.columns) + ' |')
        L.append('|' + '---|' * (len(p.columns) + 1))
        for u, r in p.iterrows():
            L.append(f'| {u} | ' + ' | '.join(f'{v:.2f}' for v in r) + ' |')
        L.append('')
    (OUT / 'summary.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
