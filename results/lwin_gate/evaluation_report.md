# Context model evaluation (results\lwin_gate)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| free_free | 3 | +2.97 | +0.55 / +0.66 / +0.79 | +1.28 / +0.79 / +0.58 | +0.01 / -1.55 / +0.12 | +0.51 | 0.27 |
| free_free_w03 | 3 | +0.92 | +0.41 / +0.66 / +0.63 | +1.11 / +0.79 / +0.55 | +0.04 / -1.55 / +0.02 | +0.51 | 0.27 |
| none_free | 3 | +0.00 | -0.04 / +0.66 / -0.18 | -0.01 / +0.79 / +0.09 | +0.09 / -1.55 / +0.15 | +0.42 | 0.34 |
| shared_free | 3 | +0.68 | +0.41 / +0.66 / +0.26 | +0.95 / +0.79 / +0.23 | +0.04 / -1.55 / +0.29 | +0.55 | 0.36 |
| shared_free_w03 | 3 | -7.01 | +0.26 / +0.66 / +0.14 | +1.02 / +0.79 / +0.20 | +0.16 / -1.55 / -0.04 | +0.53 | 0.36 |
| trialvalue_free | 3 | +3.21 | +0.00 / +0.66 / -0.23 | +0.01 / +0.79 / +0.04 | +0.59 / -1.55 / +0.12 | +0.58 | 0.35 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| free_free | +0.55 / +0.65 | +0.30 / +0.34 | +0.01 / +0.49 | +0.51 / +0.99 |
| free_free_w03 | +0.51 / +0.65 | +0.28 / +0.34 | +0.00 / +0.49 | +0.51 / +0.99 |
| none_free | -0.10 / +0.65 | +0.21 / +0.34 | +0.11 / +0.49 | +0.43 / +0.99 |
| shared_free | +0.29 / +0.65 | +0.27 / +0.34 | +0.23 / +0.49 | +0.55 / +0.99 |
| shared_free_w03 | +0.17 / +0.65 | +0.25 / +0.34 | -0.07 / +0.49 | +0.53 / +0.99 |
| trialvalue_free | -0.13 / +0.65 | +0.19 / +0.34 | +0.08 / +0.49 | +0.60 / +0.99 |
