# Value-belief fits (srsClassic, two-target choice trials; RT >= 80 ms)

## 08-19: 75 choice trials
- none  : NLL 48.8, AIC 103.6; b0 +1.322, s +0.424, lapse +0.500
- oracle: NLL 46.5, AIC 101.0; b0 +68.830, s -53.298, lapse +0.500, beta +20.000
- delta : NLL 48.1, AIC 106.2; b0 +0.581, s +0.143, lapse +0.000, beta +2.147, alpha +0.010
- mix   : NLL 48.0, AIC 106.0; b0 +0.802, s +0.282, beta +10.083, alpha +0.010, w +0.248
- beliefs from mix fit (alpha 0.010)
- delta belief ranks the rich identity correctly on 87% of choice trials
- single-target rt_s vs belief of shown identity: Spearman +0.026 (p=0.72, n=184)
- single-target sac_peak_velocity vs belief of shown identity: Spearman -0.084 (p=0.25, n=184)

## 08-20: 228 choice trials
- none  : NLL 147.5, AIC 300.9; b0 -0.556, s -0.279, lapse +0.000
- oracle: NLL 143.4, AIC 294.7; b0 -1.782, s -1.006, lapse +0.500, beta +1.001
- delta : NLL 140.1, AIC 290.2; b0 -4.084, s -2.415, lapse +0.500, beta +20.000, alpha +0.010
- mix   : NLL 142.6, AIC 295.3; b0 -0.845, s -0.424, beta +8.626, alpha +0.010, w +0.324
- beliefs from delta fit (alpha 0.010)
- delta belief ranks the rich identity correctly on 92% of choice trials
- single-target rt_s vs belief of shown identity: Spearman -0.209 (p=0.00026, n=301)
- single-target sac_peak_velocity vs belief of shown identity: Spearman +0.020 (p=0.73, n=301)


## Reading
- 08-20: beliefs from a slow delta rule (alpha at the 0.01 bound; still ranks the rich identity correctly on 92% of
  choice trials) explain choices only as a MIXTURE: value-driven and near-deterministic on ~1/3-1/2 of trials (mix w 0.32;
  delta-with-lapse at the 0.5 lapse bound), side/salience-biased otherwise. A noisy-belief account (graded, unbounded
  belief noise) is not what the data prefer. Single-target RT falls with the shown identity's belief (rho -0.21,
  p = 3e-4), so value reaches the saccade system on single-target trials.
- 08-19: no evidence of value use in choices (M_none best by AIC); single-target RT/velocity unrelated to belief.
- For the models: per-trial beliefs in `<session>_trials_with_belief.csv`; weak behaviour route = mixture/lapse
  (plus side bias), with belief-noise and weak-weighting routes kept as alternatives in the nuisance grid.
