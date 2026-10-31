# Does SC circuitry resolve competition between task factors? What our data say (2026-10-31)

Question (James): SC selects targets by a winner-take-all mechanism thought to rely on inhibition within and between
the colliculi. Does SC resolve conflict between goal-directed factors (reward, probability) and stimulus-driven
factors (salience, face identity) through its own dynamics (resolver), or does it carry out an outcome decided in
loops through basal ganglia and thalamus (hub)?

Plans: `plans/2026-10-31_competition_plan.md`, `plans/2026-10-31_intercollicular_plan.md` (both independently
reviewed). Details: `results/competition/NOTE_P1_P2.md`, `p3_interactions/`, `p4_timing/`, `results/intercollicular/`.
Bilateral results: bilateral-SC-recordings session notes in `D:\Google Drive\HermanLab\data\Newton_SC_2026\docs\`.

## Short answer
These data show that SC carries both factor classes, combines the goal-directed factors with each other the way
behaviour does, and shows map-wide competitive dynamics after a target appears. They do not show where conflict
between goal-directed and stimulus-driven factors is resolved, and no analysis of the existing data can: the
4factors task has one target per trial, and the bilateral conflict task is underpowered and offers no early
window in which the two factor classes compete inside SC. Inhibition that ignores factor identity, which is what
the literature and our data are consistent with, does not imply a hub: a Trappenberg-style field can resolve
goal-vs-stimulus conflict with location-only inhibition if the factors arrive as location-specific drive.

## What we found
1. Two factor classes enter SC at different times, on different axes.
   - Stimulus-driven: image vs bullseye and salience change the first visual response (onset 50-70 ms, peak
     70-80 ms) and are aligned across neurons (r = 0.40): one stimulus-strength axis. Face vs non-face comes later
     (onset ~130-150 ms, peak ~200 ms).
   - Goal-directed: the reward cue raises pre-target activity (+0.31 sp/s); the location prior raises it a little at
     the expected location (+0.24 sp/s) and suppresses the expected target's sustained response (expectation
     suppression, onset ~110-120 ms).
   - Goal and stimulus effects are not aligned across neurons (|r| <= 0.12): no single "priority" axis.
2. SC combines the goal-directed factors as behaviour does.
   - Behaviour: the reward cue speeds saccades to the expected location and slows them to the unexpected opposite
     location (Feynman reward x opposite +17.9% RT, p < 1e-6; peak velocity mirrors it).
   - Classic SC neurons: the reward cue enlarges the visual response to an expected target and shrinks it to an
     unexpected one (reward x expectation +0.27 SD, p = 7e-6; putative interneurons +0.08, p = 0.07), peaking at
     80-100 ms, i.e. as a gain set before the target appears.
   - Goal x stimulus interaction is weak: the reward gain is slightly smaller for images than bullseyes
     (-0.08 SD, p = 0.03).
3. Competitive spatial dynamics are present.
   - When the target appears away from the expected location, the prior's head start at the expected location is
     cancelled within ~100 ms, and activity at every non-target location drops below baseline from ~150 ms.
   - Pre-saccadic suppression is tuned to direction: strongest for the 180-deg opposite target in both putative
     interneurons (33% vs 19% at 90 deg) and classic neurons (12% vs 3%).
   - These are the dynamics a map-wide winner-take-all predicts, but inputs that update at target onset predict them
     too.
4. SC state predicts behaviour, but not location-specifically. Higher pre-target activity predicts faster saccades
   whether the neuron's field is at the target, at the expected competitor, or elsewhere (-5 to -6% RT per SD):
   general readiness, not a competition between locations.
5. Local circuit.
   - Putative interneurons are at least as factor-selective as classic neurons; their ipsilateral responses look
     externally driven (no hemifield boundary, no relay latency, not saccade-locked, salience but not reward).
   - Candidate excitatory connections (sharp 1-4 ms CCG peaks, 1% of pairs) link neurons with similar selectivity
     (location, face/non-face). Inhibitory troughs are too rare (8) to test whether inhibition is factor-specific.
6. Between the colliculi: no evidence that the other SC drives ipsilateral responses; cross-SC coupling is flat at
   the spike-timing and single-trial levels. Sparse strong commissural inhibition is excluded; weak inhibition
   converging from hundreds of neurons is not.
7. Models: the fitted RNNs reproduce location tuning but not identity or salience selectivity, get the probability
   sign wrong, and cannot identify the route of the reward-context input; their weights cannot decide the question.

## How this bears on resolver vs hub
- Consistent with both: both factor classes are present in SC before the saccade; goal factors are combined with
  each other in SC; spatial competition is visible.
- Leaning hub-with-local-execution (weakly): goal-directed signals act as gains set before the target (pre-target
  reward and prior; reward x expectation at the visual-response peak), rather than being built up by SC dynamics
  after target onset; stimulus- and goal-driven effects sit on separate axes with only a small interaction. This is
  what a circuit receiving pre-weighted goal signals would show, but a resolver could also use pre-set gains.
- Not supported by our data: factor-specific inhibition inside SC; SC state deciding between locations; a relay of
  competition through the other colliculus.
- Implication for the project's core hypothesis: "unstructured I->E weights support external loops" is not a valid
  inference on its own; factor-agnostic inhibition is compatible with SC resolving the conflict.

## Experiments that would decide it
1. Conflict task with luminant targets, no fixed delay (time pressure) and a variable onset asynchrony between the
   salient and the rewarded stimulus. Resolver: the outcome depends on when each signal arrives in SC within its
   competitive window. Hub: the outcome tracks the goal signal's arrival from the loop regardless of SC timing.
2. Caudal-dorsolateral SNr inactivation (muscimol) while recording SC in the conflict task. Hub: reward's influence on
   choice collapses while SC's spatial competition still runs. Resolver: reward still competes via other inputs.
3. Local SC GABA-A antagonism (bicuculline) in the conflict task. Resolver: choice proportions change in a
   conflict-dependent way. Hub: execution deficits only.
4. Simultaneous bilateral SC recordings with mirrored, interleaved, luminant targets (enough classic units per side)
   to test commissural competition directly.
