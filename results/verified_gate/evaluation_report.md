# Context model evaluation (results\verified_gate)

Effects: pre-target reward/block context (switch-triggered, sp/s), visual probability component (sp/s; negative = weaker response at the frequent location), spatial (pref - opposite, sp/s). 'r' = correlation across recorded neurons between model and recorded effect (pooled over sessions, mean over seeds). Delta LL = held-out Poisson log-likelihood per neuron per bin relative to `none_free` (x1000).

| Variant | Seeds | Delta LL x1000 | Reward pre: model / rec / r | Block pre: model / rec / r | Prob vis: model / rec / r | Spatial r | I->E share of reward-context current |
|---|---|---|---|---|---|---|---|
| free_free | 3 | -17.43 | +0.51 / +0.37 / +0.61 | +1.32 / +1.06 / +0.61 | +0.15 / -1.30 / -0.11 | +0.49 | 0.31 |
| free_free_w03 | 3 | -21.86 | +0.42 / +0.37 / +0.62 | +0.98 / +1.06 / +0.58 | +0.10 / -1.30 / -0.05 | +0.49 | 0.31 |
| none_free | 3 | +0.00 | +0.00 / +0.37 / -0.07 | +0.17 / +1.06 / +0.15 | +0.03 / -1.30 / +0.02 | +0.65 | 0.33 |
| none_free_nowin | 3 | -128.33 | -0.08 / +0.37 / +0.03 | +0.11 / +1.06 / +0.07 | +0.03 / -1.30 / +0.03 | +0.35 | 0.38 |
| none_free_nowin_nodrift | 3 | -119.73 | +0.04 / +0.37 / -0.02 | -0.11 / +1.06 / +0.08 | +0.04 / -1.30 / +0.05 | +0.35 | 0.42 |
| shared_free | 3 | -14.50 | +0.28 / +0.37 / -0.05 | +1.23 / +1.06 / +0.34 | +0.22 / -1.30 / -0.15 | +0.55 | 0.38 |
| shared_free_w03 | 3 | -19.64 | +0.15 / +0.37 / -0.04 | +1.15 / +1.06 / +0.34 | +0.14 / -1.30 / -0.12 | +0.50 | 0.38 |
| trialvalue_free | 3 | -11.64 | +0.00 / +0.37 / -0.18 | +0.14 / +1.06 / +0.15 | +0.50 / -1.30 / +0.09 | +0.56 | 0.35 |

Held-out check: r between model effect and the recorded effect on test trials only, and the ceiling r(recorded non-test trials, recorded test trials).

| Variant | Reward pre r / ceiling | Block pre r / ceiling | Prob vis r / ceiling | Spatial r / ceiling |
|---|---|---|---|---|
| free_free | +0.35 / +0.61 | +0.39 / +0.59 | -0.07 / +0.07 | +0.48 / +0.98 |
| free_free_w03 | +0.37 / +0.61 | +0.33 / +0.59 | -0.13 / +0.07 | +0.49 / +0.98 |
| none_free | +0.01 / +0.61 | -0.02 / +0.59 | -0.04 / +0.07 | +0.63 / +0.98 |
| none_free_nowin | +0.04 / +0.61 | +0.02 / +0.59 | -0.00 / +0.07 | +0.35 / +0.98 |
| none_free_nowin_nodrift | -0.01 / +0.61 | +0.05 / +0.59 | -0.02 / +0.07 | +0.35 / +0.98 |
| shared_free | -0.03 / +0.61 | +0.26 / +0.59 | -0.08 / +0.07 | +0.54 / +0.98 |
| shared_free_w03 | -0.06 / +0.61 | +0.29 / +0.59 | -0.13 / +0.07 | +0.49 / +0.98 |
| trialvalue_free | -0.16 / +0.61 | +0.03 / +0.59 | +0.05 / +0.07 | +0.55 / +0.98 |
