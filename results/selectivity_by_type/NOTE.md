# Visual-epoch ROC selectivity by monkey and cell type (grid_v13, free_free)

`scripts/compare_selectivity_by_type.py`, then `scripts/summarize_selectivity_by_type.py` (table in
`grid_v13_free_free_summary.txt`). Selectivity = 2 x (ROC area - 0.5), 50-250 ms after target onset, at the
preferred location, all verified trials within each neuron's stable period; location is preferred vs opposite
(preferred from even trials, scored on odd). Each RNN unit is the unit fit to that neuron (types always match).
RNN rates are Poisson-sampled and each of 5 noise draws is scored separately, so RNN and recorded values carry
comparable sampling noise; 6 restarts. Significance: 500 label permutations, two-sided p < 0.05 (5% by chance).
Signs: positive = face, high salience, high reward, high probability, preferred location higher.

Feynman: 79 putative interneurons (I), 40 classic SC (E). Newton: 111 I, 63 E.

% significant, recorded -> RNN (mean over restarts):

| Factor | Feynman I | Feynman E | Newton I | Newton E |
|---|---|---|---|---|
| Location | 61 -> 35 | 57 -> 29 | 66 -> 48 | 78 -> 58 |
| Identity | 9 -> 5 | 10 -> 4 | 14 -> 5 | 11 -> 6 |
| Salience | 28 -> 16 | 15 -> 11 | 14 -> 6 | 17 -> 6 |
| Reward | 9 -> 11 | 10 -> 12 | 17 -> 28 | 11 -> 23 |
| Probability | 20 -> 22 | 5 -> 18 | 22 -> 43 | 10 -> 43 |

## Findings
1. Location: RNN units keep which neurons are spatially tuned (per-neuron r 0.83-0.90 in every group) and the
   recorded I/E ordering within each monkey, at about 60-70% of the recorded magnitude; strength varies a lot
   across restarts.
2. Identity and salience: RNN units are at chance in all groups. Recorded neurons are weakly selective
   (9-28% significant; a small non-face preference; high salience higher), so the models do not carry these
   selectivities although both are model inputs.
3. Reward: RNN selectivity matches or exceeds recorded (per-neuron r 0.35-0.63), most so in Newton.
4. Probability: recorded neurons respond less on high-probability trials (signed -0.02 to -0.05); RNN units
   respond more (+0.05 to +0.13) and in Newton are significant twice as often as recorded. The models do not
   reproduce the probability effect (as in the context report).
5. Caveat: RNN observation noise is Poisson, while recorded variability is super-Poisson, so the RNN percent
   significant is if anything inflated; shortfalls (location, identity, salience) are not a noise artifact.
