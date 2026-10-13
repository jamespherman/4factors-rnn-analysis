# Recurrent weight structure vs recorded selectivity (cond)

Seeds: [42, 43, 44]. Sessions combined per seed with Stouffer's method; z-scores against 1000-permutation nulls (targeting: 250).

| Block | Factor | Test | z (seed 42) | z (seed 43) | z (seed 44) |
|---|---|---|---|---|---|
| E→E | identity | like | -1.04 | -1.21 | +1.69 |
| E→E | identity | targeting | +0.27 | +0.70 | +0.88 |
| E→E | salience | like | -0.15 | -0.65 | +2.89* |
| E→E | salience | targeting | -0.87 | -0.76 | +1.33 |
| E→E | reward | like | -0.32 | -0.34 | +0.55 |
| E→E | reward | targeting | +1.02 | +2.17* | -1.45 |
| E→E | probability | like | -0.42 | -0.30 | +1.20 |
| E→E | probability | targeting | +0.54 | +1.19 | +1.33 |
| E→E | location | spatial | +1.66 | +0.55 | +0.64 |
| I→E | identity | like | +0.16 | -1.00 | -0.59 |
| I→E | identity | targeting | +0.88 | +1.10 | +1.45 |
| I→E | salience | like | -1.96* | -1.10 | -1.13 |
| I→E | salience | targeting | +0.15 | +0.64 | -0.53 |
| I→E | reward | like | -0.84 | +0.16 | +0.11 |
| I→E | reward | targeting | +1.19 | +0.70 | -1.60 |
| I→E | probability | like | +1.11 | +1.83 | -0.87 |
| I→E | probability | targeting | +0.69 | +0.80 | -0.46 |
| I→E | location | spatial | +0.28 | -2.03* | -0.52 |
| E→I | identity | like | -1.29 | +1.67 | +0.91 |
| E→I | salience | like | +0.46 | -0.73 | -0.66 |
| E→I | reward | like | +1.02 | +0.41 | -1.61 |
| E→I | probability | like | -0.65 | +0.67 | -0.40 |
| E→I | location | spatial | -1.09 | +1.20 | +0.73 |
| I→I | identity | like | +1.10 | +2.45* | +0.98 |
| I→I | salience | like | +1.21 | -2.18* | -1.30 |
| I→I | reward | like | +1.33 | +0.65 | +2.04* |
| I→I | probability | like | -1.52 | +0.76 | +0.53 |
| I→I | location | spatial | +0.02 | +2.43* | +1.08 |

\* p < 0.05 uncorrected; \*\* q < 0.05 after Benjamini-Hochberg across all tests within a seed.

## Cross-seed correlation of recorded-unit weights

| Session | E→E | I→E | E→I | I→I |
|---|---|---|---|---|
| 08_13 | 0.20 | 0.13 | 0.00 | 0.31 |
| 08_14 | 0.10 | -0.01 | 0.01 | -0.03 |
| 08_15 | 0.08 | 0.05 | 0.01 | -0.04 |
