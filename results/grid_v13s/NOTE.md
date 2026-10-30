# Context grid on the 2026-10-30 exports (grid_v13 interleaved, grid_v13s switch split)

Same design as grid_v12/grid_v12s (8 variants × 7 sessions × 6 restarts each), on the exports after
curation round 2 and the new SC classic rule: 293 neurons (103 E, 190 I; v12: 116 E, 154 I).
Newton 08_13 uses the 2026-10-30 re-export with fixed event timing. 30 grid_v13s runs were trained
on CRC (PyTorch 2.5.1), the other 642 on GCP A100 spot VMs (PyTorch 2.9; `slurm/gcp_fleet.sh`,
`slurm/gcp_startup.sh`; about $77). Evaluation sharded (`evaluate_context_models.py --shard/--merge`).
Report: `results/context_report_v13.html`.

## Reward pre-target effect, all sessions (held-out r, mean over 6 restarts)

| Variant | Interleaved (v13) | Switch split, test windows (v13) | Same, v12 |
|---|---|---|---|
| shared | 0.38 | 0.06 | 0.21 |
| sharedI | 0.32 | 0.13 | 0.19 |
| freeI | 0.38 | 0.21 | 0.19 |
| freeE | 0.39 | 0.17 | 0.19 |
| free | 0.45 | 0.26 | 0.16 |
| free_qshift | 0.21 | 0.16 | 0.12 |
| shared_qshift | – | 0.17 | 0.10 |
| none | 0.15 | 0.14 | 0.14 |
| *ceiling* | 0.53 | 0.20 | 0.21 |

Paired over restarts, switch split, test windows (mean ± SE, restarts > 0):

| Difference | v13 | v12 |
|---|---|---|
| shared − free | −0.20 ± 0.08 (2/6) | +0.04 ± 0.05 |
| free − free_qshift | +0.10 ± 0.06 (4/6) | +0.04 ± 0.06 |
| shared − shared_qshift | −0.11 ± 0.04 (1/6) | +0.11 ± 0.08 |
| free − none | +0.11 ± 0.06 (5/6) | +0.02 ± 0.03 |
| sharedI − freeI | −0.08 ± 0.05 (2/6) | −0.00 ± 0.03 |
| held-out LL, shared − free (×1e-3) | −3.8 ± 4.7 | +2.7 ± 3.2 |

## Findings
1. The interleaved split still leaks, but less: the shifted control reaches 0.21 of per-unit's 0.45
   (v12: 0.35 of 0.47).
2. On held-out switches per-unit context now scores higher than shared context (−0.20 ± 0.08) and
   somewhat above its shifted control (+0.10 ± 0.06). Shared context scores below its own shifted control.
3. This does not identify per-unit input. With realistic teachers (`results/recovery_v12_drift`,
   v12 teachers, ceiling matched to 0.21; v13 ceiling 0.20) shared − free is −0.08 to −0.19 for every
   teacher, including the shared and no-context ones; −0.20 is within that range. Held-out LL
   (−3.8 ± 4.7) is within 1 combined SE of the per-unit teachers and 1–2 from the shared and
   no-context teachers. Conclusion unchanged: the fits cannot decide between shared and per-unit
   context input.
4. Fit quality: PSTH median r 0.65 vs median ceiling 0.58 over 293 units (v12: 0.60 vs 0.52);
   R01 location selectivity r 0.49 [0.27, 0.82] over restarts (v12: 0.57), reward 0.24.
