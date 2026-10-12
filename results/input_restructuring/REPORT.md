# Input Restructuring Experiments Report

Generated: 2026-01-29 17:39:49

## Overview

These experiments test whether SC factor selectivity arises from:
1. **Direct input drive** - neurons receive factor information directly
2. **Recurrent dynamics** - selectivity emerges from network interactions

**Total experiments:** 18
**Successful:** 18

## Experimental Conditions

| Experiment | Description | Seeds |
|------------|-------------|-------|
| E0 | Baseline: all neurons, all factors, at target onset | 1 |
| E1 | Temporal fix: spatial factors from trial start | 1 |
| E2 | Primary: RF-matched spatial (E only), sparse features (p=0.5) | 5 |
| E3a | Sparsity test: p=0.25 | 3 |
| E3b | Sparsity test: p=0.75 | 3 |
| E4 | Replication with different seeds | 5 |

## Fit Quality

| Experiment | Mean Val Corr | Std | vs E0 |
|------------|---------------|-----|-------|
| E0 | 0.1646 | 0.0000 | 100% |
| E1 | 0.1763 | 0.0000 | 107.1% |
| E2 | 0.1585 | 0.0279 | 96.2% |
| E3a | 0.1649 | 0.0121 | 100.2% |
| E3b | 0.1703 | 0.0113 | 103.4% |
| E4 | 0.1235 | 0.0169 | 75.0% |

**Success criterion:** E2 achieves >= 80% of E0's correlation

**PASSED:** E2 (0.1585) >= 80% of E0 (0.1317)

## Emergent Selectivity

Emergent selectivity = factor selectivity in neurons that do NOT receive that factor as input.

| Factor | E2 Mean |sel|| E2 Frac Sig | Ablation Reduction |
|--------|---------|-------------|---------------------|
| reward | 0.2058 | 50.0% | 24.7% |
| probability | 0.3207 | 84.3% | 47.8% |
| salience | 0.1565 | 33.8% | -328.1% |
| identity | 0.1993 | 40.7% | -271.9% |

## Key Findings

### 1. Fit Quality Preserved?

E2 achieves 96.2% of E0 correlation.
**Result:** Input restructuring does NOT substantially degrade fit quality.

### 2. Emergent Selectivity Detected?

- **reward:** 50.0% of non-receivers show significant selectivity
- **probability:** 84.3% of non-receivers show significant selectivity
- **salience:** 33.8% of non-receivers show significant selectivity
- **identity:** 40.7% of non-receivers show significant selectivity

**Result:** Yes, neurons show selectivity for factors they don't receive directly.

### 3. Recurrence Dependence?

- **reward:** 24.7% reduction after full recurrence ablation
- **salience:** -328.1% reduction after full recurrence ablation

**Target:** >= 50% reduction confirms recurrence dependence.

### 4. Cross-Seed Consistency?

Mean pairwise I->E weight correlation: r = 0.007

**Result:** I->E weight structure varies across seeds.

## Figures

- `fit_quality_comparison.png` - Validation correlation across experiments
- `emergent_selectivity_by_factor.png` - Fraction significant per factor
- `ablation_results.png` - Selectivity before/after ablation
- `cross_seed_consistency.png` - I->E weight correlation across seeds
- `ie_weight_structure.png` - I->E weight heatmap and distribution

## Interpretation

If emergent selectivity is detected AND ablation substantially reduces it,
this supports the hypothesis that factor selectivity arises from local
recurrent dynamics involving interneurons, rather than direct input drive.

## Next Steps

1. Compare I->E weight structure to E neuron factor selectivity
2. Test whether specific I->E connection patterns predict selectivity
3. Validate findings with additional sessions/animals
