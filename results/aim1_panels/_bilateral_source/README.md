# Bilateral source tables (intermediate, for B6/B7/B9 builders)

CSV exports of MATLAB tables saved by the Newton_SC_2026 peer repo (`D:\Google Drive\HermanLab\data\Newton_SC_2026`,
read-only; srsClassic bilateral recordings 08-19 and 08-20), written by `scripts/aim1_panels/export_bilateral_tables.m`
(2026-11-01). Nothing is recomputed: each file is a table from a saved .mat:
- `competition_choice_tc_<session>.csv`, `competition_choice_trials_<session>.csv` <- `data/bilateral/competition_choice.mat`
- `competition_normalisation_units.csv`, `competition_normalisation_sims.csv` <- `data/bilateral/competition_normalisation.mat`
- `onset_estimators_S1.csv` <- `data/qc/onset_estimators.mat`
- `intercollicular_ccg_split_R.csv` <- `data/bilateral/intercollicular_ccg_split.mat`
- `intercollicular_bound_B.csv` <- `data/bilateral/intercollicular_bound.mat`
- `intercollicular_latency_combos_<act|supp>.csv`, `intercollicular_latency_units_<act|supp>.csv` <-
  `data/bilateral/intercollicular_latency.mat`
