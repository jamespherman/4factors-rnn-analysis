# P4 — bilateral SC state (Newton, simultaneous left + right SC, sessions 08-20 and 08-19)

**a.** P(chose right) by tercile of the pre-target left-minus-right SC population bias (300 ms before target onset),
mean +/- binomial SE. 08-20: 0.20 / 0.37 / 0.53 (n = 228 choice trials, z = 3.78); 08-19: 0.50 / 0.64 / 0.77
(n = 77, z = 2.32). z = logistic-regression z of the bias beyond previous side, running side mean, reward side, gaze x
and eye movements; 0/189 and 0/38 circular shifts as large (p < 0.0053, p < 0.026).

**b.** Pooled cross-midline CCG during fixation, (observed - jitter expectation) / expectation, 1-ms bins,
-40..+40 ms; positive lag = right-SC spike after left. Shaded: the 2-8 ms commissural bands. 08-20: 473 pairs,
1,149 segments, 651 s; 08-19: 1,212 pairs, 876 segments, 435 s. Band modulation (band_fit): 08-20 L->R -0.12%
(95% CI -1.2 to +0.9), R->L +0.19% (-0.9 to +1.3); 08-19 L->R -0.94% (z -1.42), R->L -0.02%. Nothing significant.

**Not plotted, in stats.csv:** pre-target fluctuations are positively shared across the colliculi (not push-pull):
partial r(left, right) = +0.155 on 08-20 (n = 598, p = 0.011; null 2.5-97.5% -0.10 to +0.12) and +0.144 on 08-19
(n = 291, p = 0.091). Source fields `r_partial`, `p2c`, `n2c`, `rNull` of `pretarget_bias.mat`.

## Files
- `P4_bilateral_state.png` (300 dpi), `.pdf` — the figure, 2 panels at about 2.4 x 2.0 in.
- `P4_a_choice_by_bias_tercile.csv` — session, tercile, n_trials, p_chose_right, se_binomial.
- `P4_b_ccg_fixation.csv` — session, lag_ms, observed, jitter_expected, modulation (fraction).
- `P4_b_bands_shaded.csv` — shaded band edges (ms).
- `stats.csv` — session, statistic, value, z, p, p_text, ci_lo, ci_hi, n, test, note. Band-fit p values are
  two-sided normal from the saved z (derived here); everything else is the saved value.
- `plot_P4.py` + `style.py` — standalone plot from the CSVs only (`python plot_P4.py`).

## Provenance (peer repo, read-only; nothing recomputed)
- `D:\Google Drive\HermanLab\data\Newton_SC_2026\data\bilateral\pretarget_bias.mat` (saved 2026-10-26 12:00;
  `code/run_pretarget_bias.m`). Terciles tabulated from saved `bias(idx2a)` and `y2a` exactly as
  `code/plot_bilateral_bias.m` panel A (MATLAB prctile = numpy `hazen`); values match the peer doc.
- `D:\Google Drive\HermanLab\data\Newton_SC_2026\data\bilateral\crossmidline_coupling.mat` (saved 2026-10-26 13:15;
  `code/run_crossmidline_coupling.m`), fields `out.<session>.fixation.{ccg, expected, lags, z, mod, mod_se}`, as
  `code/plot_bilateral_coupling.m` panel A.
- Results text: `docs/PLAN_2026-10-26_bilateral_hub_resolver.md` (Analysis 2, 2c, 3) and `docs/STATUS.md`
  ("2026-10-26 — bilateral hub-vs-resolver analysis"). Original figures: `figures/bilateral/pretarget_bias.pdf`,
  `figures/bilateral/crossmidline_coupling.pdf`.
- Builder: `D:\Projects\4factors-rnn-analysis\scripts\aim1_panels\build_R01_P4.py` (2026-11-01).
