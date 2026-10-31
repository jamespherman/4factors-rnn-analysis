# srs — hub vs resolver bilateral SC models for the srsClassic task

Model code for plan v2 (`D:\Projects\4factors-rnn-analysis\plans\2026-10-31_hub_resolver_models_plan.md`).
Owner: conflict-task-rnn-f8. Training and analyses: 4factors-RNN (GCP/CRC; copy the repo, do not edit in place).
Dependencies: torch, numpy, scipy, pandas. Run from `conflict-task-RNN/code/` (package import `srs`).

## Architectures (`SRSConfig.arch`)

| | Upstream (BG-like) inputs | Upstream output → SNr | SC extra inputs | Where identity × belief × location is bound | Where the choice is resolved |
|---|---|---|---|---|---|
| **B** priority hub | fixation, belief [V1,V2], identity-at-location (delayed `up_latency`) | graded, independent per location: `dis_i = dis_max·sigmoid(o_i)` | — | upstream | SC (pools race / WTA) |
| **C** outcome hub | same as B | competitive: `dis = dis_max·softmax(c_beta·[o_L,o_R,o_nogo])[:2]` | — | upstream | upstream (near-categorical) |
| **A** resolver | fixation, belief (no identity → no conjunction possible) | graded, location-blind | identity-at-location + belief via FIXED random weights onto all SC units | SC | SC |

Shared SC (all): 2 colliculi × [E_pool 40 | E_other 40 | I_local 10 | I_comm 10], Dale's law, τ 50 ms, dt 10 ms,
commissural I → other SC (p 0.3, toggle `commissural`). Pool i (in the SC opposite location i) receives a weak visual
input (salience gain `sal_gain`, default 0.1 = isoluminant) and SNr inhibition `w_snr·SNr_i`,
with `SNr_i = relu(snr_tonic + snr_fix·fixation − dis_i)` (fixed tonic level; fixation held inside SC).
Readout: independent per-location hazard `sigmoid(g·(pool_i − θ))`, first event wins, an event before go = fixation
break. No normalisation in the readout.

Nuisance routes (per trial in the table, or config): `engaged` (mixture/lapse: 0 → both identities get the mean belief),
`side_bias` (locus `bias_locus` = "SC" or "upstream"), belief noise (supply noisy V1/V2), `slow_shared` / `slow_side`
(slow OU noise onto the pools; pre-target state), `noise_sc`, `noise_up`.

## Trial table (the interface)

Columns: `trial_type` (0 single, 1 choice), `id_L`, `id_R` (1 T1, 2 T2, 0 absent), `sal_L`, `sal_R`, `rew_L`, `rew_R`
(ms if chosen), `V1`, `V2` (belief ms), `side_bias`, `fix_dur`, `delay` (target→go ms; negative = go first),
`onset_L`, `onset_R` (asynchrony), `engaged`. Extra columns pass through.

- Recorded sequences: `srs.task.from_recorded(csv)` converts the bilateral-SC-recordings / 4factors export
  (`*_trials_with_belief.csv`; side code 1 = RIGHT, 2 = LEFT, verified against saccade endpoints; failed rows dropped
  by default; the animal's choice kept as `monkey_choice_loc`, RT as `monkey_rt_ms`).
- Synthetic sessions with the recorded block structure: `make_session(cfg, rng, ...)` (+ `belief_delta_rule`).
- Training trials: `make_training_trials(cfg, n, rng, stage)`.

## Main calls

```python
from srs import SRSConfig, build_handbuilt, simulate, behavior_summary
from srs.task import from_recorded
from srs.train import load_net

table = from_recorded(".../08-20_trials_with_belief.csv")
net = build_handbuilt("B", SRSConfig(device="cpu", slow_side=0.3))     # Stage 0 hand-build (A, B or C)
# net = load_net("results/srs/B_s0/model.pt", device="cuda")            # trained network
res = simulate(net, table, seed=0, perturb=None)
res["table"]   # input columns + outcome (0 completed,1 fixbreak,2 noresp,3 inaccurate), choice_loc, chosen_id,
               # reward_ms, rt (ms from go), go_step, onset_step_L/R
res["rates"]   # sc [n,T,200], pool [n,T,2], snr [n,T,2], dis [n,T,2] (model units); res["t"] ms from fixation onset
res["units"]   # sc_side (L/R colliculus), type (E_pool/E_other/I_local/I_comm), field, is_exc, pool_loc
behavior_summary(res)
```

Perturbations (`simulate(..., perturb=...)`):
`{"bicuculline": {"sc": "L", "d": 0.3, "scope": "side"|"pool", "snr": True}}` (scales local+commissural I→E and SNr→E
GABA-A at the site), `{"muscimol": {"sc": "R", "d": 1.0}}`, `{"comm_cut": True}`,
`{"silence": {"sc": "both", "types": ["I_comm"]}}`, `{"snr_value_clamp": True}` (value-independent upstream signal).

## Training (GPU)

```
cd conflict-task-RNN/code
python -m srs.train --arch B --seed 0 --name B_s0 --out /path/results/srs [--set commissural=false bias_locus=upstream ...]
```
Curriculum: S1 single-target flat reward (hold fixation, saccade after go) → S2 real values, belief = block means →
S3 noisy beliefs (`belief_noise`). `stage_iters` default 1500,1500,1500, batch 256. Outputs `config.json`, `log.json`,
`model.pt` (state_dict + config).

## Calibration knobs added 2026-10-31 (requests from 4factors-RNN)
- `a_input_targets` = "all" | "E" | "E_other": which SC units receive A's fixed identity/belief input (trained A).
  The hand-built A already routes them only to E_other conjunction units.
- C reaction time: hand-built param `c_gain` (default = `dis_max`; lower = slower), `cfg.c_dis_gain` for trained C;
  `cfg.dis_tau` (ms) low-pass on the upstream disinhibition for all architectures (0 = off).
- RT resolution: `cfg.rt_substep` (default True; event time uniform within its 10-ms step) and `cfg.rt_jitter_sd`
  (ms of Gaussian motor jitter). Outcome classification uses the event step, so jitter never creates fixation breaks.
- `cfg.shape_vis` (default 0): purely visual bar-orientation preference of the visually driven SC E units (the E pool
  at the target's location): drive × relu(1 + shape_vis·u_k·sign(identity)), u_k ~ N(0,1) fixed per unit (separate
  RNG stream, so all other weights are unchanged), sign +1 for T1 and −1 for T2; same in A, B, C, same time course
  as the visual input. Gives every architecture a block-invariant identity code. Non-pool E units get no visual input.
- Training-time nuisance randomisation (stages listed in `nuisance_stages`, default ("S3",)): per batch
  w ~ U(`train_engaged_min`, 1) and engaged ~ Bernoulli(w) per trial; side_bias ~ U(−`train_side_bias_max`, +max) per
  trial. Defaults (1.0, 0.0) = off. Side bias always enters through FIXED weights (onto the pools for
  bias_locus="SC", onto the upstream per-location outputs for "upstream"), so training cannot unlearn it.
- RT pressure: `time_cost` (reward units per ms of RT, completed trials, training only; e.g. 5e-4).
- `engaged` is filled with 1 wherever a table lacks it (bug fix: training trials lacked the column).

## Status (2026-10-31)
- Hand-builds B, C, A: hold fixation, saccade after go (RT 140–220 ms), choices follow value; graded behaviour via
  slow pool noise / belief noise / side bias verified on CPU. C needs larger upstream noise to become stochastic.
- Training loop: CPU smoke test (a few iterations) passes for A, B, C. Full training not yet run (no local GPU):
  run a pilot (1 seed per arch) and check that S1 reaches high completion before trusting S2/S3.
