# Plan: are putative SC interneurons sources or targets of inhibition between the two colliculi? (v2)

Date: 2026-10-31. v1 was reviewed by an independent instance (review summary at the end); this v2 adopts most of it.
Background research: `plans/2026-10-31_research_intercollicular.md`.

## Hypotheses (for ipsilateral responses and suppression in one recorded SC)
- H-external (default null): feedforward/extra-collicular input (RF edge, FEF/LIP/pulvinar bilateral fields,
  SNr/ZI inhibition) produces ipsi responses and suppression; no role for the other SC.
- H-target: the response is relayed from the opposite SC (commissural excitation / inhibition).
- H-local: non-classic neurons are local interneurons driven by own-SC neurons (not exclusive with H-target).
- H-source (commissural projection neurons): cannot be tested with one-SC data; only "consistent with" using
  depth (deep layers) and waveform as weak priors. Stated as such.
Known constraints: cross-hemisphere spike-count correlations ~0 (Katz, Yu, Herman & Krauzlis 2023); bilateral
Newton 2026 pooled cross-SC CCG flat (fraction x depth >= 0.04 detectable at 85%), slow pre-target coupling +0.15.

## Measures used throughout (all 293 neurons; class only as covariate)
- Continuous ipsi/contra index per window: visual (50-250 ms after target), pre-saccadic (-50..0 ms re
  saccade onset), post-saccadic (0..100 ms). Response = rate minus pre-target baseline.
- Flags: pause-type (saccade-epoch suppression at >= 3 of 4 locations, fixation-like), depth relative to the
  session's visuomotor (classic, saccade-active) units, N15 vertical-meridian units, waveform width.

## Phase 1 - decisive local tests on unilateral data (today, ~6 h)
G1. Distance-matched geometry. Targets are 90 deg apart in polar angle. For each neuron with a contra preferred
    location P, compare responses at the two 90-deg neighbours: one contra (across the horizontal meridian) and
    one ipsi (across the vertical meridian), and at the 180-deg opposite (ipsi). Caveat: the SC map ends at the
    vertical meridian, so an RF-tail account also predicts contra-neighbour > ipsi-neighbour; the diagnostic parts
    are (a) suppression below baseline at ipsi locations (RF tails cannot suppress) and (b) elevation specificity
    (cat anatomy: commissural excitation to the same elevation, inhibition to the opposite elevation).
G2. Latency-amplitude calibration. From classic neurons' contra responses at several amplitudes (preferred vs
    neighbour location x salience x identity), fit visual-onset latency as a function of response amplitude.
    Ask whether non-classic ipsi responses lie on the curve (feedforward-like) or are systematically later (> ~5 ms;
    relay-like). Onset latency from spike times (1 ms), bootstrap CIs.
G3. Alignment test. On ipsi-target trials, is each neuron's modulation better explained aligned to target onset or
    to saccade onset (cross-validated single-trial variance explained under each alignment, using RT spread)? A
    relay of the opposite SC's motor burst should be saccade-locked and start ~20 ms before saccade onset.
G4. Factor inheritance. Does a neuron's ipsi-location modulation by reward, salience, identity and probability match
    (in sign/magnitude, possibly inverted) the classic population's modulation at the mirrored contra location?
    Under H-target the opposite SC carries our mirror-location factor pattern; feedforward visual input would
    carry salience/identity but not necessarily blocked reward/probability. Permutation null over factor labels.
G5. Controls in all of the above: pre-saccadic windows end at saccade onset (no reafference); report gaze-position
    sensitivity; check suppression is not block-specific (reward/probability blocks); depth as covariate;
    with and without the 6 N15 meridian-only units.

## Phase 2 - bilateral data (requests to the bilateral-SC-recordings session, sent today)
C1. Latency test (decisive if ipsi responses exist): in MG blocks and conflict single-target trials, onset of side
    Y's ipsiversive-saccade modulation relative to side X's contraversive burst (condition-averaged and per trial).
    Y starting at/before X's burst rules out a relay.
C2. Population-source C2: X's full population (all units or MUA) predicting Y's single-trial responses, targets
    grouped by a continuous ipsi index recomputed in MG (recovers "can't tell" units); PSTHs and slow drift removed.
C3. Power-to-hypothesis bound: translate the flat pooled CCG into an upper bound on per-pair commissural inhibition
    and compare with what H-target needs to produce the ipsi suppression we see in the unilateral data.
C4. Class/ipsi-index-resolved CCGs only as a secondary check; DLAG only for which side leads on choice trials.

## Phase 3 - CCGs and coupling (unilateral; ~half day)
B3. Within-SC jitter-corrected CCGs (validated `src/ccg.py`): E->I peaks onto non-classic units (H-local), as a
    fraction of pairs normalised by per-pair power, with inter-unit distance as covariate; ipsi-responsive
    non-classic targets as a planned contrast. Troughs reported but not interpreted (expected near zero).
B2. Rate-matched single-trial coupling to own-SC classic latents (FA after drift removal), secondary.
B4. Descriptive correlations of ipsi index with depth, waveform, coupling and E->I evidence (no clustering unless
    clear bimodality).

## Phase 4 - models (only if Phases 1-2 leave H-target viable)
A-check. Sensitivity of the project's central I->E conclusion to the labels: refit/reanalyse with ipsi-responsive
    non-classic units removed from the I pool or given an explicit ipsi input. If unchanged, the commissural
    question matters little for the central hypothesis.
A-GLM. Poisson GLM per neuron: contra/ipsi stimulus kernels by distance from the RF centre, own-SC population
    activity (single trial), mirror template, saccade kernels, drift. Cheap, identifiability checkable.
A-pilot (go/no-go, 1-2 sessions, ~30 runs): two-SC mirror-twin RNN (tied within-SC weights, mirrored inputs,
    independent noise, learned cross E/I matrices), teachers local / crossI / external-mimic, inputs full vs
    contra-only, plus the nested control (fixed mirrored contra-PSTH input with learned weights, and a
    time-shuffled version). Framed around whether the second SC fixes the current failures (probability-effect
    sign, identity/salience selectivity) relative to a capacity-matched one-SC model. Scale up only if the
    confusion matrix is clearly diagonal and the twin beats the fixed template.

## Decision rules
- If G1-G4 explain ipsi responses as RF-extent/feedforward (on-curve latency, target-locked, no factor
  inheritance, no suppression), stop Phase 4 except A-check, even if other analyses show "mirror structure".
- If C1 shows Y modulation at/before X's burst, H-target is excluded for that response type.
- No connectivity interpretation from any model without a passing identifiability pilot.

## Outputs, compute
results/intercollicular/<analysis>/ with NOTE.md each; summary report; commits on restore-lost-work.
Phases 1-3 local CPU. Phase 4 pilot on GCP (< $20). Running GCP total ~ $77 of $500.

## Review summary (v1 -> v2)
Adopted: H-external as default; H-source untestable with one SC; continuous ipsi index instead of class contrasts;
pause-type flag; geometry (distance/elevation) tests; latency-amplitude calibration, alignment and factor-inheritance
tests instead of lag regression (2-7 ms lags not resolvable from natural bursts); rate-matching in coupling;
CCG power normalisation and distance covariate; no clustering; Part A gated by a pilot with an external-mimic
teacher and a fixed-template nested control, aimed at the project's known model failures; A-check of the central
I->E conclusion; revised bilateral requests (latency, population source, power bound).
Not adopted: microsaccade regression in Phase 1 (eye data at 25 ms only in the export; deferred to a later check
from raw eye traces if a result depends on delay-period activity).
