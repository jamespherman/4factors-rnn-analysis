# Window-mean loss gate, verified trials (2026-10-13)

Successful trials with a verified PLDAPS match only, with the reward input at the true delivery
time. 3 sessions x 3 seeds; full table in `evaluation_report.md`.

- **The window-mean loss is essential on clean data.** Without it (`none_free_nowin`), held-out
  LL is 0.12 per neuron per bin lower and spatial r drops from 0.65 to 0.35.
- **Drift inputs no longer matter** (`nowin` vs `nowin_nodrift`): the slow drift was largely the
  changing mix of failed trials.
- **Per-unit context (`free_free`) reproduces both pre-target context effects held-out:** reward
  +0.35 / ceiling 0.61, block +0.39 / ceiling 0.59.
- **Shared context reproduces block (+0.26) but not reward (-0.03) per neuron.**
- **Visual probability effect:** on clean test trials its per-neuron ceiling is only 0.07, so only
  its population mean (-1.30 sp/s) can be tested. No model reproduces it.
- **Fits vary by seed, mostly in 08_15** (spatial r 0.04-0.98 there; 0.5-0.8 in 08_13/08_14).
  Each seed also drew its own split, so seeds aren't comparable. Next (`results/verified_grid`):
  a fixed split (`--split-seed 0`), 6 restarts, and the best restart chosen by selection-set LL.
