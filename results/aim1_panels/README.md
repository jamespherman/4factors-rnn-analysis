# Aim 1 panels: figures + self-contained data (2026-11-01)

Each folder holds one write-up item (`results/AIM1_FINDINGS_WRITEUP.md`, v3): `<ITEM>.png` (300 dpi) and `.pdf`
(editable text), tidy CSVs with every plotted point, band and CI, `stats.csv` (every quoted statistic with n, test,
window, unit class, session), `plot_<ITEM>.py` (standalone: reads only the CSVs in its folder plus `style.py`; edit it to
restyle; run `python plot_<ITEM>.py` from anywhere), and `README.md` (panels, column dictionary, provenance, what hub and
resolver predict). Builders that compute the CSVs from the analysis outputs: `scripts/aim1_panels/build_<ITEM>.py`.
Plan and review: `plans/2026-11-01_aim1_panels_plan.md`.

Colours (all panels): recorded data near-black; resolver (A/AE) orange, dashed when a prediction; priority hub (B) blue;
outcome hub (C) aqua; null bands light grey.

| Item | Folder | Data | What it shows | Hub vs resolver |
|---|---|---|---|---|
| B1 | `B1_reward_x_prior` | 4factors | RT/peak velocity by location class x reward cue: a crossover in Feynman | Behaviour; no architecture contrast (additive null drawn) |
| B3 | `B3_factor_alignment` | 4factors | Factor effects across units (E, I), null and reliability ceiling | Tests the SC priority-map prediction; reward too unreliable per unit to test |
| B4 | `B4_map_competition` | 4factors | Expected-location PSTH when the target is elsewhere; suppression by angle | Compatible with both |
| B5 | `B5_rt_state_specificity` | 4factors | RT vs pre-target rate by RF category, competition half-plane | Against local competition setting RT |
| B6 | `B6_bilateral_behaviour` | srsClassic | Choice vs reward difference; RT vs value with drift controls | Behaviour (RT-value effect does not survive drift controls) |
| B7 | `B7_lr_signal_normalisation` | srsClassic | L-R signal: choice yes, reward difference no; no competitor normalisation | Trained resolvers also show neither: not discriminating |
| B8 | `B8_ipsi_latency_relay` | 4factors | Amplitude-corrected ipsi relay excess vs 10-27 ms band | Against a slow commissural relay |
| B9 | `B9_intercollicular_relay` | srsClassic | X burst vs Y ipsiversive onset; flat cross-SC CCG vs relays | Inconclusive on timing; dense relay not excluded |
| B10 | `B10_ccg_pair_selectivity` | 4factors | Sharp-peak pairs: similarity connected vs unconnected; troughs | Not discriminating (inhibitory case untestable, n = 5) |
| B11 | `B11_rnn_selectivity_scatter` | 4factors | Recorded vs data-fitted RNN ROC selectivity (figure moved here) | Fitted weights cannot locate the computation |
| C1 | `C1_rt_value_signature` | models + srsClassic | S9 per network (all architectures), drift-controlled monkey values | Graded vs categorical delivery, not resolver vs hub |
| D1 | `D1_bicuculline_dose_response` | models | In-silico bicuculline/muscimol dose-response | No architecture differs |

`_bilateral_source/`: intermediate CSV exports of the bilateral peer repo's saved MATLAB tables (B6, B7, B9 builders).
srsClassic items (B6, B7, B9, C1 recorded values) come from the two bilateral srsClassic sessions.
