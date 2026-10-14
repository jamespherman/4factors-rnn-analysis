# Window-mean loss gate, all exported trials

> **Superseded (2026-10-13):** run on all exported trials, including fixation breaks. Kept only to compare with `results/verified_gate`.

Result (3 sessions x 3 seeds; full table in `evaluation_report.md`):

| Variant | Reward pre, held-out r / ceiling | Block pre | Prob vis | Spatial |
|---|---|---|---|---|
| none_free | -0.10 / 0.65 | +0.21 / 0.34 | +0.11 / 0.49 | +0.43 / 0.99 |
| shared_free | +0.29 / 0.65 | +0.27 / 0.34 | +0.23 / 0.49 | +0.55 / 0.99 |
| free_free | +0.55 / 0.65 | +0.30 / 0.34 | +0.01 / 0.49 | +0.51 / 0.99 |
| trialvalue_free | -0.13 / 0.65 | +0.19 / 0.34 | +0.08 / 0.49 | +0.60 / 0.99 |

- The window-mean loss passes the gate for the pre-target context effects: `free_free` gets close to the ceiling.
  It also improves spatial fit (0.43-0.60 vs 0.26-0.41 without it).
- `shared_free` lies between `none_free` and `free_free` on reward context. That is the comparison that
  bears on H_ext vs H_local.
- No model reproduces the visual-epoch probability effect, not even `trialvalue_free` (+0.59 sp/s vs -1.55 recorded).
