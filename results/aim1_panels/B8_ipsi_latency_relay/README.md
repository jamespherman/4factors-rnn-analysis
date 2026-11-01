# B8: ipsilateral visual responses show no reliable commissural relay delay

Write-up item B8 (`results/AIM1_FINDINGS_WRITEUP.md`). Unilateral 4factors recordings.
Regenerate the figure: `python plot_B8.py` (reads only the CSVs here and `style.py`).
Rebuild the CSVs: `scripts/aim1_panels/build_B8.py` (repo git commit 6355c0a, branch restore-lost-work; under 1 min).

## Panels
- **a (main): relay excess.** For each unit and stimulus class (high-salience bullseye, low-salience bullseye,
  image) with a significant visual activation both at the ipsilateral 90-deg neighbour and on the contralateral
  side: onset latency at the ipsi neighbour minus contralateral onset latency, minus the latency difference expected
  from the amplitude difference (calibration slope -14.1 ms per ln unit, panel c). Left: every condition (n = 111;
  41 units; filled = non-classic / putative interneuron, open = classic). Right (zoom): median +3.7 ms with 95% CI
  -2.2 to +7.1 ms (unit-level cluster bootstrap). Orange band: 10-27 ms, the "commissural relay prediction".
- **b (secondary): raw latencies**, ipsi vs contra, identity line, relay band drawn as y = x + 10..27 ms. Not
  amplitude-corrected: ipsi responses are weaker, so they are expected to be later by a median of 2.7 ms from
  amplitude alone. Show only with that caveat.
- **c: the calibration** used for the correction: classic units, contralateral activations (preferred and contra
  90-deg neighbour locations x 3 stimulus classes), n = 220 conditions from 58 units, OLS latency = 134.6 - 14.1 ln(amp),
  r = -0.34, residual SD 35 ms.

## What hub vs resolver predict
- If ipsilateral responses were relayed from the other SC (the commissural route a two-SC resolver would use), they
  should be delayed relative to direct contralateral responses beyond what their weaker amplitude explains.
- The band 10-27 ms is what the bilateral planted-relay simulation (S1) read as the Y-minus-X lag with the rise
  estimator for activations (`D:\Google Drive\HermanLab\data\Newton_SC_2026\docs\NOTE_2026-10-31_intercollicular_for_4factors.md`,
  lines 31 and 43). The planted synaptic delays themselves were 2, 5 and 10 ms (`B8_bands.csv`, row
  `planted_synaptic_delay`; not drawn). **Caveat:** that band was derived for a different estimator (saccade-aligned
  population onset in bilateral data), so it is an approximate, not a calibrated, prediction for this within-unit
  half-peak measure. A short (2-7 ms) synaptic relay is not excluded by this CI (PHASE1_NOTE.md says the same).
- An external (feedforward visual) source, the hub view, predicts excess about 0. The data: median +3.7 ms, CI
  includes 0 and excludes the 10-27 ms band; 16% of single conditions fall inside the band.

## Statistics (`stats.csv`)
- n = 111 is **conditions** (unit x stimulus class), not units: 41 units (classic 20 conditions, non-classic 91).
  The contralateral reference is the contra 90-deg neighbour when that response was significant (82), otherwise the
  preferred location (29, not equidistant).
- Median raw difference +5.0 ms (Wilcoxon p = 0.016); expected from amplitude +2.7 ms; excess +3.7 ms (Wilcoxon
  p = 0.22). These reproduce `results/intercollicular/g23_timing/summary.md` exactly.
- CI of the median excess (recomputed; the original CI [-1.3, 6.8] was not saved and its seed/method could not be
  reproduced exactly): unit-level cluster bootstrap (resample the 41 units with all their conditions; 10,000
  resamples; percentile; seed 0): **[-2.2, +7.1] ms**. Variants: condition-level bootstrap (ignores repeated units)
  [-1.0, +6.6]; unit bootstrap that also refits the calibration slope on resampled calibration units [-2.4, +8.0].
  Median of per-unit medians: +2.5 ms (n = 41, Wilcoxon in stats.csv).
- Response rates "27% vs 25%": `results/intercollicular/g1_geometry/summary.md`, block `## visual window`, row
  `I both n=154`, **activated** at ipsi-nb 27% vs contra-nb 25% (non-classic units, 6 diagonal-target sessions,
  paired test vs pre-target baseline p < 0.01). Recomputed from `g1_geometry/rows.npy`: 27.3% vs 24.7%. The same row
  gives suppression 19% (ipsi) vs 26% (contra).

## Column dictionary
- `B8_pairs.csv` (one row per plotted condition): session, unit_id (cluster id), unit_type (E classic / I
  non-classic), stim_class (bull_hi, bull_lo, image), contra_reference (cn = contra 90-deg neighbour, P = preferred),
  lat_ipsi_ms, lat_contra_ms (half-peak onset, ms after target onset), amp_ipsi_sp_s, amp_contra_sp_s (peak evoked
  rate, sp/s), lat_diff_ms (ipsi - contra), ln_amp_ratio (ln(amp_ipsi / amp_contra)), expected_from_amp_ms
  (slope x ln_amp_ratio), excess_ms (lat_diff - expected; panel a).
- `B8_calibration.csv`: classic contralateral conditions of panel c (amp_sp_s, lat_ms, location_role P/cn).
- `B8_calibration_fit.csv`: the fitted line (amp_sp_s, fit_lat_ms).
- `B8_bands.csv`: band, low_ms, high_ms, source (relay prediction, planted delays, CI of the median).
- `stats.csv`: statistic, value, n, test, window, unit_class, session_filter, ci_low, ci_high.

## Provenance and method
- Source script `scripts/ic_g23_timing.py` (commit de99e41) -> `results/intercollicular/g23_timing/rows.npy`;
  `scripts/ic_g1_geometry.py` -> `g1_geometry/rows.npy`. Verified trials (v13 exports), each unit's stable trials,
  >= 20 trials per condition; 6 diagonal-target sessions (Feynman 08_12-08_15, Newton 08_13-08_14; Newton 08_15 has
  meridian targets and is excluded).
- Onset: target-aligned PSTH (1 ms bins, Gaussian SD 5 ms) minus the -100..0 ms baseline; first time >= 30 ms that the
  evoked rate reaches half its 30-250 ms peak and stays there 10 ms; amplitude = that peak (>= 5 sp/s required).
