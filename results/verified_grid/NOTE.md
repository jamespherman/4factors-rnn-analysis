# Context grid, verified trials (2026-10-14)

Successful trials with a verified PLDAPS match only, with true reward times. Window-mean loss on
(`--lwin-scale 1`), context-conditioned losses off. One fixed train/select/test split
(`--split-seed 0`), 6 restarts per variant and session. Full tables are in `evaluation_report.md`.

## Fit quality is now good
- Spatial r averages 0.76-0.86 across restarts. The best restart per session, chosen by
  selection-set LL, reaches 0.92-0.98 (ceiling 0.97).
- On all exported trials without the window loss it was 0.28. The improvement comes from the
  trial filter together with the window-mean loss.

## Reward context per neuron: only per-unit context input reproduces it

Held-out r with the recorded pre-target reward-context effect (ceiling 0.64):

| Variant | Mean over restarts | Restarts | Best restart |
|---|---|---|---|
| free_free (per-unit context weights) | +0.43 | +0.30 to +0.52 | +0.48 |
| shared_free / _global / _spatial | +0.04 / -0.01 / -0.03 | -0.23 to +0.36 | -0.05 / -0.09 / +0.16 |
| sharedI_free / sharedE_free | +0.03 / -0.04 | -0.20 to +0.28 | -0.05 / -0.09 |
| trialvalue_free | +0.00 | -0.14 to +0.19 | -0.03 |
| none_* (no context) | -0.07 to -0.01 | | |

A shared context signal, whether into E, I or both, and whether the I->E weights are free or
constrained, does not produce the neuron-specific pattern of reward-context modulation in these
fits. Per-unit context input does, at about 70% of the ceiling.

Caveats:
- Failing to learn is not the same as being unable to. The recovery test (Stage D) has to show
  that a shared-context student recovers a teacher whose neuron-specific context effects arise
  through its I->E weights. Until then this is a negative result about what these fits found.
- The per-unit model also has more input parameters. Its held-out LL is no better than the
  others (delta LL -9.3 mean, -2.5 best restart), so the extra parameters are not paying off
  in likelihood. They matter only for this one small effect.

## Other results
- **Block context per neuron:** every variant reaches about +0.2 (ceiling 0.46), including no
  context. No variant separates from the others.
- **Visual probability effect:** the per-neuron ceiling is 0.12, so the clean test trials can't
  test it. No model reproduces its population mean (-1.30 sp/s).
- **I->E constraints:** held-out LL differences between free, spatial and global I->E weights
  are within about +/-5 x1000, the same size as restart-to-restart spread. One `none_global`
  restart failed (-3506). The data do not demand a particular I->E structure.
- **Reward-context current:** I->E carries 27-39% of the pre-target reward-context current to
  E units in every variant.

## Added 2026-10-14: per-unit context to I only or E only

Held-out r with recorded reward context (ceiling 0.64), 6 restarts each:

| Variant | Mean | Range |
|---|---|---|
| free_free (all units) | +0.43 | +0.30 to +0.52 |
| freeI_free (I units only) | +0.36 | +0.18 to +0.45 |
| freeE_free (E units only) | +0.36 | +0.11 to +0.54 |
| shared_free | +0.04 | -0.20 to +0.36 |

- **Per-unit context into interneurons alone, relayed through the I->E weights, reproduces the
  neuron-specific reward-context pattern** about as well as per-unit input to every unit. What
  fails is a single shared context signal, not the local route.
- **These fits can't tell whether the neuron-specific input lands on E or on I units.** They
  indicate only that it must arrive already specific to individual neurons, not as one global
  reward state.
- The recovery pilot (`results/recovery_pilot`) tests whether this E-vs-I distinction is
  identifiable at all.
- Held-out LL: freeI -0.2 and freeE +1.4 x1000 vs none_free (restart means). Same size as the
  restart spread.

## Context-shift controls (2026-10-14)
- `*_shift50` (context circularly shifted by half a session) is **not a valid control**. The reward map
  switches exactly every 48 successful trials, so it is periodic, and a half-session shift leaves it
  correlated with the true reward map: r = -0.67 (08_13), +0.83 (08_14), +0.17 (08_15). Free per-unit
  weights can use an anti-correlated input as easily as a correlated one. Its results (reward r 0.36
  all-unit, 0.29 I-only) are kept for the record only.
- `*_qshift` (`--context-shift-trials 24,48`) shifts each channel by a quarter of its own period,
  which leaves it uncorrelated with the true context (|r| <= 0.08; 0.2 for the block in 08_13).
  This is the control used in the report.
