# Spatial profile of classic SC neurons and putative interneurons (v13 exports)

`scripts/sc_spatial_profile.py`: reads the pipeline's cached screening tests (processed `session_data.mat`,
`cached_friedman_results`, method paired_ttest_stable4f_v1, computed 2026-10-28 to 10-30) and reconstructs the
classic rule of `screen_sc_neurons.m` (2026-10-28). Output: `summary.txt`, `location_profiles.png/pdf`.
Rule (c)'s Welch half-of-contra test is not recomputed; "ipsi activation" below is a significant ipsilateral
activation, which fails (c) for every interneuron that has no other flag.

## Findings
1. Why putative interneurons are non-classic (Feynman n=79 / Newton n=111): ipsilateral activation alone
   32% / 27%; ipsilateral activation + visual/saccade suppression 28% / 26%; suppression alone 15% / 23%;
   no contralateral activation (alone or combined) 25% / 24%.
2. Activation pattern over the 4 task locations: interneurons respond in both hemifields 59% / 53%,
   contra only 15% / 23%, ipsi only 10% / 7%, suppression only 15% / 16%. Classic neurons are contra only
   95% / 83% (the rest pass (c) with ipsi < half contra).
3. Visual-epoch location ROC of interneurons is similar in the contra-only and both-hemifield groups
   (0.63-0.66), so "responds in both hemifields" does not mean "not spatially selective": the ipsi response is
   usually much smaller (median visual contra-ipsi index +0.52 in both monkeys).
4. Classic neurons with weak visual location ROC (< 0.6; Feynman 27/40, Newton 20/63) have small visual
   responses (median best-contra evoked 1.9 sp/s vs 4-10 sp/s for the others) and are mostly saccade-related;
   their saccade-epoch contra-ipsi index is +0.39 / +0.66. They are spatially selective, at the saccade.
5. Newton 08_15 targets are at 90/180/270/360 deg; for a right SC only 180 counts as contralateral, so the
   vertical-meridian targets count as ipsilateral. 19 of the 24 interneurons there with ipsi activation have it
   only at the vertical meridian; 6 of 36 are non-classic only because of that.
