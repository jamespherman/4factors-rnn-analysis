# In-silico bicuculline in SC: predictions for resolver (A/AE), priority hub (B), outcome hub (C) (2026-10-31)

## Question
Which outcome of a unilateral SC bicuculline injection (Hikosaka & Wurtz 1983/1985) during srsClassic distinguishes the
architectures? Bicuculline blocks GABA-A at the site: local/commissural I -> E and nigral (SNr) -> SC inhibition.

## Logic
- B (priority hub): block value reaches SC as graded SNr disinhibition. Blocking SNr -> SC at the site removes the
  channel that carries value for targets in the injected field ("occlusion"): on single-target trials, RT (and SC
  activity) for injected-field targets should lose their dependence on belief value; the opposite field keeps it.
- A/AE (resolver): value enters SC through excitatory inputs and is bound in SC's recurrent dynamics. Blocking GABA
  disinhibits the site (faster, more saccades to that field) but the value dependence of RT should largely persist.
- C (outcome hub): single-target RT has no value dependence before the injection; the injection adds a side bias.
- Choice trials: all architectures predict a bias toward the injected field. The diagnostic is whether the choice still
  depends on the injected-field target's value (B: reduced; A: preserved) and on the other target's value.

## Simulation
- Networks: hand-builds A, B, C (stage0 calibration: w, bias, C c_gain/theta) and the trained v2 ensemble A_s0-5,
  AE_s0-5, B_s0-5, C_s0/4/5 (engaged w 0.65, bias 0, an uncalibrated nominal setting; the CRC calibration is pending).
- Trial table: the 08-20 recorded sequence with fitted beliefs (from_recorded), plus synthetic sessions (make_session,
  8 blocks) to increase trial counts; 5 noise seeds per condition.
- Conditions: control; bicuculline at LEFT SC (affects RIGHT-field targets, location 1), d in {0.5, 0.9}, with SNr
  blockade (realistic) and local-only (snr=False; dissociates the two GABA sources); E units of the whole side.
- Measures (per network, pooled over seeds):
  1. Validation (Hikosaka-Wurtz): fixation-break rate and direction; single-target RT for injected vs other field.
  2. Occlusion: Spearman rho(RT, belief value of the shown identity) and the OLS slope (ms per 100 ms value) on
     single-target trials, injected field vs other field, control vs injection. Index = slope_inj / slope_ctrl.
  3. Choice: logistic regression of chose-right on (V_right - V_left) and sal_right, and separately on V_right and
     V_left; control vs injection: intercept (bias) and value slopes.
  4. Pool activity: mean delay-window pool rate vs value for the injected-field pool (neural occlusion).
- Report per architecture: median and range over networks; flag networks with runaway activity (fixation breaks >50%).

## Caveats planned
Trained v2 RT is too fast (110-140 ms); predictions are qualitative/directional. Spread of the drug is modelled as the
whole side's E units at fraction d. The model's SNr is the only value-carrying GABA channel in B; real SC also has
zona incerta and other GABAergic inputs.

## v2 after independent review (incorporated)
1. Full 2x2 dissociation: local-only (I -> E) and SNr-only (SNr -> E) at matched dose alongside the combined drug.
   SNr-only is the clean channel test (B loses value dependence for the injected field; A should not).
2. Floor/saturation: dose series (d 0.05-0.35; d >= 0.9 makes every network break fixation toward the injected field
   on every trial, the Hikosaka-Wurtz irrepressible-saccade regime, so high doses are uninformative). Primary neural
   measure is the delay-window pool rate vs value (before the hazard nonlinearity); compare architectures at matched
   behavioural effect (RT shift / fixation-break rate), not only at matched d.
3. Fixation breaks reported (rate, direction, dependence on value), not silently dropped.
4. A's upstream: value information in the upstream disinhibition quantified per network (Spearman of delay-window
   dis with shown value); snr_value_clamp control (value-free upstream) run for every network.
5. Claim scoped to resolver vs hubs; B vs C is not separable by the drug (both route selection through SNr).
6. SNr operating range reported (fraction of the delay at the relu floor). Expected occlusion index for B under
   SNr-only at dose d: about 1 - d (value and tonic components scale together).
7. Muscimol (d 0.5) as the excitability-matched comparison. Engaged w = 1 for all networks (mechanism prediction);
   calibration caveat stated. Decision rule: SNr-only (d 0.5) pool-slope index < 0.6 = value via SNr (B-like),
   > 0.8 = value not via SNr (A-like); ranges over networks reported.
8. Real-world caveat: low doses; bicuculline also blocks local inhibition; a nigrotectal-terminal-selective
   manipulation (e.g. optogenetic/chemogenetic silencing of SNr terminals in SC) is the cleaner version.
