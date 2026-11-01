# P6: ipsilateral activation, task-trained local SC model vs monkey SC (model-to-monkey comparison)

Regenerate: `python plot_P6.py` (reads `P6_monkey.csv`, `P6_model.csv`, `style.py`). Builder:
`scripts/aim1_panels/build_R01_P6.py`.

**Model.** Bilateral SC E-I networks (80 E + 20 I per colliculus, Dale's law, inhibitory-only commissure) trained by
reinforcement learning on the Conflict Task (conflict-task-RNN; `docs/results_summary.md`,
`docs/model_predictions_for_Aim1.md` Prediction 2, written 2026-09-30, before this comparison). Values: fraction of
units activated (Wilcoxon vs -100..0 ms baseline, p < 0.01, 50-150 ms after target onset) by a single target at any
of 4 contralateral / any of 4 ipsilateral locations, mean +- SD over networks (10 trained; 5 with sparse
cross-midline E -> I excitation added). Source tables: `conflict-task-RNN/figures/probe_main_tables.md`,
`probe_commEI_tables.md`. The I bars average commissural and local-only I units.

**Monkey.** 4factors single-target recordings (Feynman, Newton; diagonal-target sessions; meridian session excluded),
classic neurons (n = 72) and putative interneurons (n = 154). Activated: paired t vs pre-target baseline, p < 0.01,
visual window, at any contralateral location tested (preferred or 90-deg contralateral neighbour) / any ipsilateral
location tested (90-deg ipsilateral neighbour or 180-deg opposite); Wilson 95% CI. Source:
`results/intercollicular/g1_geometry/rows.npy` (scripts/ic_g1_geometry.py).

**Result.** Contralateral activation matches (interneurons 49% [41, 57] vs model I 45-62%). Ipsilateral activation does
not: 33% [26, 40] of putative interneurons (Feynman 32%, Newton 33%) vs 2-4% of model I units (7-16% even with
cross-midline excitation added); classic neurons 15% [9, 25] vs model E 0%. The model's ipsilateral test is the more
lenient (4 locations vs 2), so the gap is conservative.

**Interpretation.** A circuit optimised to perform the conflict task locally does not produce bilaterally activated
interneurons; the monkey's are activated by ipsilateral targets at the same latency as by contralateral ones (no
relay delay; B8). Their ipsilateral drive therefore comes from input outside the local SC circuit, consistent with
SC receiving bilateral, externally computed signals (hub). Caveats: different tasks (model: Conflict Task; monkey:
4factors single targets); the I/E mapping assumes putative interneurons correspond to model I units.
