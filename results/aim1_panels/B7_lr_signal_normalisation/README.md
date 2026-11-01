# B7: reward difference and choice in the left-minus-right SC signal; competitor normalisation (Newton, srsClassic)

These are **srsClassic** bilateral recordings (Newton, both SCs simultaneously), choice trials with one target per
hemifield, session 08-20 (primary; 228 choice trials, 227 with all covariates). 08-19 values are in the CSVs
(descriptive). Units: all recorded units of both SCs (no E/I split).

Figure: `B7_lr_signal_normalisation.png` / `.pdf`; regenerate with `python plot_B7.py` (reads only the CSVs here).

## Panels
- **i** Time course (50-ms windows re target onset; placebo windows before target; pre-saccadic -100..0 ms
  reference at right) of
  - added choice information: z = sign(added CP - 0.5) x sqrt(LR), where LR is the likelihood-ratio chi2 (1 df) of
    adding the window's L - R signal to the covariate model M0 (salience diff, reward diff, previous choice, gaze, pre-
    target L - R bias); added CP = AUC of the window signal residualised on M0;
  - reward difference: OLS t of reward difference in W ~ reward diff + salience diff + choice + pre-target bias.
  - Grey band: null at the Bonferroni level over the 11 windows (two placebo + nine post-target): |z| < 2.84,
    |t| < 2.87 (df 222). Uncorrected 1.96/1.97 levels are in `B7_i_null_bands.csv`.
  - Orange: trained resolver networks' (A and AE, 12 networks) S1 = t of the reward-difference coefficient in the
    model L - R signal (delay window 100-450 ms: band = range over networks, dashed = mean; pre-saccadic: one open
    circle per network). Each network value is the mean of 10 simulated sessions at the recorded unit and trial counts
    (`results/hub_models/sig_v2/<net>/draws.csv`; `scripts/srs_stage0.py`). Hub networks' S1 is in the CSV.
- **ii** Per-unit competitor-reward coefficient (Poisson GLM log-rate change per 100 ms of the ipsilateral
  target's reward; model (a): + contra reward + block + pre-target count), delay epoch +300..+467 ms, 26 units.
  Strip below (same x axis): recorded unit mean (+0.004); null distribution of the mean under pure choice coding
  (200 simulations, 2.5-97.5%: -0.046 to +0.037); planted normalisation exp(-0.1 per 100 ms), i.e. -10%/100 ms
  (distribution of the mean, power 1.00 delay / >= 0.98 all 08-20 epochs); the trained networks' S2 (mean competitor
  coefficient, per network: resolvers orange, priority hubs blue, outcome hubs green).

## Do the trained resolvers predict these effects? No
- S1 (reward-difference t, delay): resolvers mean +0.18 (range -0.87 to +1.09 over 12 networks); hubs about 0. Pre-
  saccadic: resolvers +0.27 (-0.88 to +0.89). No resolver network approaches the Bonferroni level; the recorded t is
  within the same range.
- S2 (mean competitor coefficient): resolvers -0.006 (range -0.023 to +0.005); hubs -0.003 to +0.005. The planted
  detectable effect is -0.10; the resolvers' values sit inside the recorded null (2.5th percentile -0.046).
- So the write-up sentence "A resolver comparing values in SC predicts both effects; a recipient of a pre-weighted
  signal predicts neither" is **not supported** by the trained networks. Suggested wording: "Neither effect is present,
  and neither separates the architectures: the trained resolver networks, simulated at these trial counts, also show
  no reward-difference coding (t about 0) and only a negligible competitor effect (-0.01 to -0.02 per 100 ms, an order of
  magnitude below the detectable -0.10). The test excludes strong divisive normalisation by the competitor's reward,
  not resolution in SC."
- Window caveat: model S1/S2 use 100-450 ms after target (delay) and the S2 model includes chose-own (like the
  recorded model (b), +0.008, null 2.5th -0.042); the recorded delay epoch is +300..+467 ms.

## Other checks against the write-up
- "added choice probability 0.60 at +100-150 ms": reproduced (0.598, p 0.0037 < 0.0045). A second window,
  +350..400 ms, also passes Bonferroni (CP 0.62, p 0.0018); pre-saccadic CP 0.80.
- "no reward difference in any window": max |t| = 1.81 (+300..350 ms); none near the 2.87 threshold.
- "power >= 0.98 for a 10% drop per 100 ms": delay power 1.00 (models a and b); 0.98 is the pre-saccadic epoch.
- "early choice window underpowered (minimum detectable AUC 0.61-0.69)": from the peer note's arithmetic (not
  recomputed); 0.69 refers to 08-19.

## Files and columns
- `B7_i_timecourse.csv`: session, window, window_kind, aligned_to (target/saccade), t_start_ms, t_end_ms, t_center_ms,
  n_trials, choice_LR, choice_p, choice_z_signed, added_CP, reward_t, reward_p_two_sided (t, df n-5),
  choice_sig_bonferroni.
- `B7_i_null_bands.csv`: session, level, alpha_two_sided, choice_z_crit, reward_t_crit.
- `B7_model_S1_S2_per_network.csv` (per network means/SDs over 10 draws) and `B7_model_S1_S2_draws.csv` (every draw):
  network, architecture (A, AE resolvers; B priority hub; C outcome hub), S1_delay_t, S1_presac_t,
  S2_mean_comp_coef (log rate per 100 ms), S2_t.
- `B7_ii_unit_coefficients.csv`: unit_row (row in the peer unit table), comp_coef_model_a, comp_coef_model_b, session,
  epoch (visual +40..+300 ms, delay +300..+467 ms re target, presacc -100..0 re saccade).
- `B7_ii_sims_of_unit_mean.csv`: session, epoch, gamma (0 = null, 0.1, 0.2 planted), model, sim, mean_comp_coef.
- `B7_ii_population_mean_summary.csv`: real mean, null mean and 2.5/97.5%, planted means and quantiles, power.
  (08-19 visual-epoch fits diverged on low-count units; ignore those rows.)
- `stats.csv`: every quoted statistic.

## Provenance
- Builder `scripts/aim1_panels/build_B7.py` (HEAD 6355c0a, 2026-11-01); MATLAB tables exported read-only by
  `scripts/aim1_panels/export_bilateral_tables.m` to `results/aim1_panels/_bilateral_source/`.
- Newton_SC_2026 (Google Drive; read-only): `data/bilateral/competition_choice.mat` (2026-10-31 11:43;
  `code/run_competition_choice.m`), `data/bilateral/competition_normalisation.mat` (2026-10-31 11:45;
  `code/run_competition_normalisation.m`); code files later edited at 12:26 to add model-run options; the recorded
  results are unchanged. Note `docs/NOTE_2026-10-31_competition_for_4factors.md` (14:19).
- Model: `results/hub_models/sig_v2/*/draws.csv` (commit 8c76cf1; 21 networks, C_s1-3 excluded as training failures).
