# B9: is ipsiversive activity a delayed copy of the other colliculus? (Newton bilateral, srsClassic)

These are **srsClassic** bilateral recordings (Newton 08-19 and 08-20), conflict single-target trials (isoluminant
targets, so responses are perisaccadic). X = the SC contralateral to the target (all units, unselected); Y = the other
SC's units whose conflict-motif call has an ipsiversive activation cell (16 units total).

Figure: `B9_intercollicular_relay.png` / `.pdf`; regenerate with `python plot_B9.py` (reads only the CSVs here).

## Panels
- **a-d** One panel per session x target side: saccade-aligned X population (grey) and Y activation units (black),
  each scaled to its perisaccadic maximum (peer `local_traces`), baseline 200 ms before target. Ticks: rise onsets
  (X grey, Y black). Orange dashed band: relay prediction = the X trace shifted later by +10 to +27 ms, the range of
  the rise-estimator read-out under the planted relay simulation S1 (true synaptic delays 2-10 ms, 5-ms kernel,
  linear/saturating/threshold transfer; the estimator reads low-rate Y units late, hence 10-27 ms). The shift is a
  display of the prediction; no simulation was rerun.
- **e** Rise-onset Delta (Y - X) per combination and the pre-stated mean over the 4 combinations (-5.5 ms; trial
  bootstrap 95% CI -8.4 to +18.3), against the S1 relay read-out band.
- **f** Pooled cross-SC CCG, 08-20 fixation (the primary bound window; 473 pairs, all units), fractional modulation
  (observed - interval-jitter expectation) / expectation per 1-ms lag, +- 1 SE (sqrt(obs)/exp). Orange: expected
  measured trough in the 2-8 ms L->R band for (dashed) a sparse relay of 50 converging X neurons (r = 5-10 / 50 =
  0.10-0.20 per spike; band = recovery 0.40-0.55) and (dotted) the strongest relay the bound does not exclude
  (r = q-bar bound 0.022-0.030, i.e. 169-462 converging neurons, "~170-500" in the write-up; its measured trough
  equals the data's lower 95% band limit, -0.012).

## Discrepancies with the write-up / peer note
- "It rises with the other side's contraversive burst (lag -6 ms, CI -8 to +18)": the mean reproduces, but the 4
  combinations are heterogeneous: +23 ms (08-19 target right), -66 ms (08-19 left; only 2 Y units), +27 ms (08-20
  right; 3 Y units), -6 ms (08-20 left; 3 units). Two of four lie inside the relay read-out band. The peer note's
  "within about 10 ms in all 4 combinations" does not match the saved onsets. Suggested wording: "On average Y's
  ipsiversive activation rises with X's burst (mean Delta -6 ms, CI -8 to +18; relay predicts +10 to +27), but the
  four session x side estimates range from -66 to +27 ms, two within the relay range, with 2-8 Y units each; timing
  neither supports nor excludes a relay."
- The 170-500 neuron bound assumes every recorded X neuron projects (f = 1) but uses the recovery factor measured at
  f <= 0.2 (0.40-0.55); the peer note gives ~0.15 at f = 1, which would loosen the bound about threefold (~60-130
  neurons). Treat "~170-500" as an upper estimate.
- "Cross-SC CCGs are flat in every group": reproduced (max |z| 1.77 over 4 Y groups x 2 windows,
  `B9_e_ccg_by_Y_group.csv`).

## Hub vs resolver
A resolver that resolves competition through commissural inhibition/relay predicts Y's ipsiversive response to
follow X's burst by the relay delay (+10-27 ms as read by this estimator; S1 simulation) and a short-latency CCG
trough; an externally driven (hub/loop) account predicts simultaneous onsets and a flat CCG. The prediction drawn is
the S1 relay of the recorded X population (peer `tests/test_onset_estimators.m`) and the back-of-envelope relay
strength of the peer `run_intercollicular_bound.m` header, not a trained-network simulation.

## Files and columns
- `B9_abcd_traces.csv`: session, target_side, x_side, y_side, y_response (act / supp), n_trials, combo_label,
  time_ms_from_saccade (1-ms steps), x_scaled, y_scaled, relay_pred_shift_lo_scaled (+10.2 ms), relay_pred_shift_hi_scaled
  (+27.3 ms). Suppression traces are included but not plotted (heterogeneous; see peer note C1).
- `B9_abcd_onsets.csv`: rise and departure onsets of X and Y (ms re saccade), delta_rise_ms, relay-predicted onset range.
- `B9_S1_relay_simulation_readout.csv`: true_delay_ms, transfer, readout_delta_rise_ms (mean), readout_sd_rise_ms.
- `B9_e_ccg_0820_fixation.csv`: lag_ms (+ = right SC after left), observed_count, expected_jitter_count,
  frac_modulation, frac_modulation_se, sparse_relay_N50_frac_mod_deep/_shallow, bound_relay_frac_mod_deep/_shallow.
- `B9_e_relay_models.csv`: parameters of the two drawn relays. `B9_e_ccg_by_Y_group.csv`: peer C4 table.
- `stats.csv`: every quoted statistic.

## Provenance
- Builder `scripts/aim1_panels/build_B9.py` (HEAD 6355c0a, 2026-11-01); tables exported by
  `scripts/aim1_panels/export_bilateral_tables.m` into `results/aim1_panels/_bilateral_source/`.
- Newton_SC_2026 (read-only): `data/bilateral/intercollicular_latency.mat` (2026-10-31 11:01;
  `code/run_intercollicular_latency.m`), `data/qc/onset_estimators.mat` (10:57), `data/bilateral/crossmidline_coupling.mat`
  (2026-10-26 13:15; `code/run_crossmidline_coupling.m`), `data/bilateral/intercollicular_bound.mat` (10:57),
  `data/bilateral/intercollicular_ccg_split.mat` (11:25); note `docs/NOTE_2026-10-31_intercollicular_for_4factors.md`.
