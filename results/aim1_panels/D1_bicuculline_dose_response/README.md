# D1 - In-silico SC bicuculline / muscimol: dose-response by architecture

Figure: `D1_bicuculline_dose_response.png` (300 dpi) / `.pdf`. Regenerate with `python plot_D1.py` (reads only the
CSVs here and `style.py`). Builder: `scripts/aim1_panels/build_D1.py`. It does no simulation and reads
`results/hub_models/bicuculline/per_network.csv`. Repo commit at build: `6355c0a` (branch restore-lost-work).

## Caption (suggested)
Unilateral bicuculline at the left SC (right field) scales GABA-A input onto its E units by (1 - d). This covers local
and commissural I -> E and SNr -> E ("full"). All four architectures and the three hand-built networks show the same
classic phenotypes:
- saccades into the injected field speed up with dose, and the other field is essentially unchanged;
- fixation breaks rise steeply at d 0.35, and almost all of them go into the injected field;
- the raw probability of choosing the injected field rises.

Muscimol (d 0.5) slows saccades into the field by +102 to +241 ms where RT is defined, and moves choice away from it in
all 24 networks. No phenotype separates resolver from hub. Magnitudes overlap, except that B (priority hub) has weaker
RT speeding (15-21 ms at d 0.2, vs 23-48 ms) and fewer fixation breaks.

## Panels
- **a, b** RT change (ms, vs the same network's control) for single-target saccades into the injected field (a) and
  the other field (b). Lines are the median over trained networks of an architecture, with the min-max range shaded.
  Diamonds are hand-built networks.
  - A median is drawn only at doses where every network of that architecture has a defined RT. RT is undefined when
    20 or fewer completed single-target saccades into the field remain.
  - At d 0.35 RT is undefined in 12 of 24 networks (A_s0, A_s1, A_s2, A_s4, A_s5, AE_s1, AE_s2, AE_s3, AE_s5, B_s5,
    C_s0, C_s5), so every trained curve stops at d 0.2.
- **c** Fixation breaks, % of all trials.
- **d** Change in raw P(choose the injected field) on two-target trials. The value-controlled logit intercept change
  (`db0`) is in the CSV and is not plotted; see the discrepancies below.
- **e, f** Muscimol d 0.5, one dot per network.
  - e: RT change. Undefined, because fewer than 21 completed saccades go into the field, in A_s0, AE_s1-AE_s4, B_s2,
    C_s0, C_s5, hand_A and hand_B.
  - f: change in P(choose injected).

## What each architecture predicts
The earlier hypothesis was that a resolver shows conflict-dependent choice changes and a hub shows execution deficits
only. It does not hold. Disinhibition compresses RTs toward a floor and biases choice in every architecture, so these
behavioural phenotypes validate the models but do not discriminate between them.

## Files and columns
- `D1_per_network_long.csv`: one row per network x condition x measure.
  - `net`, `arch`, `handbuilt`.
  - `cond`: ctrl, full05, full10, full20, full35, musc50.
  - `manipulation`, `dose_d`.
  - `source_column`: the column name in per_network.csv.
  - `measure`: one of the following.
    - `rt_change_injected_field_ms`, `rt_change_other_field_ms`.
    - `fixation_break_rate`: fraction of all trials.
    - `fixation_break_fraction_toward_injected`: NaN if 10 or fewer breaks.
    - `delta_p_choose_injected_field`.
    - `delta_logit_bias_toward_injected_value_controlled`: change in the logistic intercept b0 of choice ~ v_right +
      v_left + salience.
  - `value`: per-network mean over 3 seeds.
- `D1_architecture_summary.csv`: trained networks only, one row per measure x architecture x condition.
  - `n_networks`, `n_defined`, `median`, `min`, `max`.
  - `plotted`: True if all networks are defined. This implements the truncation rule.
- `stats.csv`: every number checked against the write-up. Columns: `statistic`, `value`, `n`, `test`, `source`,
  `note`.

## Write-up numbers vs data (also in stats.csv)
- **"d 0.2: 17-42 ms faster".** These are the architecture medians: A -33.5, AE -37.2, B -16.5, C -42.2. The
  per-network range is 15.0-47.7 ms in trained networks; B spans 15.0-20.5. Hand-built: A 56.9, B 32.2, C 16.9.
- **"other field unchanged".** True to within ~1 ms except C_s0, which is 10.9 ms faster at d 0.2 and 20.4 ms faster
  at d 0.35.
- **"choice biases toward it".** True for raw P(choose injected) in 23 of 24 networks at every dose. B_s5 moves away:
  -0.8, -3.2, -18.4 and -17.6 % points.
  - With the value-controlled logit intercept, 7-8 of 24 networks move away at d 0.05-0.2.
  - This includes all three C networks at d 0.05 and 0.1 (C_s0 -0.11/-0.13, C_s4 -1.61/-3.13, C_s5 -2.04/-9.52).
    Their raw P(choose injected) still rises.
- **"fixation breaks on 20-70% of trials" at d >= 0.35, "in all 24".** False.
  - The range is 0.3-71%.
  - Below 20%: B_s2 8.2%, B_s3 3.0%, B_s4 18.8%, C_s4 4.6%, hand_A 4.5%, hand_B 0.3%, hand_C 0.9%.
- **"99-100% toward it".** The defined range is 88-100% (hand_C 88%, B_s2 94%, B_s3 93%). It is undefined for hand_B,
  which has too few breaks.
- **Muscimol "+120-210 ms".** These are the architecture medians: A +136, AE +118, B +174, C +214. The per-network
  range is +102 to +241. RT is undefined in 10 of 24 networks, because almost no saccades go into the field.
- **"choice moves away".** True in 24 of 24 networks (-24 to -75 % points).
- **Control fixation breaks.** These are <= 0.6% except B_s5, which is at 2.1%.
