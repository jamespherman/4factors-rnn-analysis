# Recurrent weight structure vs recorded selectivity (cond_target)

Seeds: [42, 43, 44]. Sessions combined per seed with Stouffer's method; z-scores against 1000-permutation nulls (targeting: 250).

| Block | Factor | Test | z (seed 42) | z (seed 43) | z (seed 44) |
|---|---|---|---|---|---|
| E→E | identity | like | +0.04 | -1.53 | +1.53 |
| E→E | identity | targeting | +0.34 | +1.57 | +0.95 |
| E→E | salience | like | -0.09 | -3.86** | +2.50* |
| E→E | salience | targeting | +0.32 | +0.24 | +4.07** |
| E→E | reward | like | +2.50* | -0.36 | +0.66 |
| E→E | reward | targeting | +1.27 | -0.40 | -0.46 |
| E→E | probability | like | +0.25 | +0.44 | +0.63 |
| E→E | probability | targeting | +0.40 | +0.96 | +2.16* |
| E→E | location | spatial | +1.92 | +1.16 | +2.24* |
| I→E | identity | like | +0.19 | -0.28 | -0.81 |
| I→E | identity | targeting | +1.78 | +2.93** | +1.02 |
| I→E | salience | like | -1.84 | -0.04 | -0.29 |
| I→E | salience | targeting | +1.22 | +0.87 | -1.62 |
| I→E | reward | like | -0.50 | +0.97 | +0.59 |
| I→E | reward | targeting | +1.42 | +1.59 | -1.61 |
| I→E | probability | like | +1.55 | +0.97 | -1.33 |
| I→E | probability | targeting | +0.08 | +1.27 | +0.57 |
| I→E | location | spatial | +0.43 | -2.37* | -0.40 |
| E→I | identity | like | -0.81 | +0.60 | +0.36 |
| E→I | salience | like | +1.29 | +1.14 | -0.06 |
| E→I | reward | like | +0.71 | +0.98 | -1.40 |
| E→I | probability | like | -1.42 | -0.35 | -1.41 |
| E→I | location | spatial | -0.14 | +1.81 | +0.25 |
| I→I | identity | like | +1.07 | +2.28* | +1.07 |
| I→I | salience | like | +1.36 | -2.21* | -1.16 |
| I→I | reward | like | +1.31 | +0.68 | +2.17* |
| I→I | probability | like | -1.78 | +0.61 | +0.49 |
| I→I | location | spatial | -0.06 | +2.23* | +0.94 |

\* p < 0.05 uncorrected; \*\* q < 0.05 after Benjamini-Hochberg across all tests within a seed.

## Cross-seed correlation of recorded-unit weights

| Session | E→E | I→E | E→I | I→I |
|---|---|---|---|---|
| 08_13 | 0.42 | 0.25 | 0.22 | 0.27 |
| 08_14 | 0.11 | -0.00 | 0.03 | -0.01 |
| 08_15 | 0.18 | 0.06 | 0.05 | -0.06 |
