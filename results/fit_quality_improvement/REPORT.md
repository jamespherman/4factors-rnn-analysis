# Fit Quality Improvement Report

Generated: 2026-01-30
Status: **Complete**

## Executive Summary

The model improvements (attention embedding + learnable h0) increased validation correlation by **18-36%** compared to the original input restructuring experiments. However, the target of 0.35-0.40 was not reached. The improvements did preserve and strengthen the emergent selectivity findings.

## Results

### Validation Correlation Comparison

| Experiment | Original | Improved | Change | Notes |
|------------|----------|----------|--------|-------|
| E0 (baseline) | 0.165 | **0.195** | +18.7% | Attention embedding |
| E1 (temporal fix) | 0.176 | **0.221** | +25.2% | Linear embedding |
| E2 (primary test) | 0.159 | **0.216** | +36.5% | Linear embedding |

### E2/E0 Ratio (Input Restructuring Impact)

| Metric | Original | Improved |
|--------|----------|----------|
| E2/E0 ratio | 96.2% | **110.7%** |

**Finding**: With improvements, E2 actually **outperforms** E0. The input restructuring may be helping the model by providing more structured inputs.

### Training Details

| Experiment | Epochs | Best Epoch | Time |
|------------|--------|------------|------|
| E0_seed0 | 2000 | 1956 | 1:55 |
| E1_seed0 | 569 | 269 | 0:39 |
| E2_seed0 | 485 | 185 | 0:32 |

## Emergent Selectivity (E2)

Neurons showing selectivity for factors they do NOT receive as direct input:

| Factor | Non-receivers | Significant | Fraction | Mean |sel| |
|--------|---------------|-------------|----------|---------|
| Reward | 28 | 25 | **89.3%** | 0.347 |
| Probability | 28 | 23 | **82.1%** | 0.265 |
| Salience | 27 | 13 | **48.1%** | 0.251 |
| Identity | 14 | 10 | **71.4%** | 0.313 |

**Comparison to original E2 results:**
- Reward: 89% vs 50% (improved)
- Probability: 82% vs 84% (similar)
- Salience: 48% vs 34% (improved)
- Identity: 71% vs 41% (improved)

## Ablation Analysis (E2)

Testing whether emergent selectivity depends on recurrence:

### Full Recurrence Ablation (W_rec = 0)

| Factor | Original |sel| | Ablated |sel| | Reduction |
|--------|-------------|-------------|-----------|
| Reward | 0.346 | 0.152 | **56.1%** |
| Probability | 0.265 | 0.159 | **39.9%** |
| Salience | 0.252 | 0.548 | -117.4% |
| Identity | 0.312 | 0.470 | -50.4% |

### I->E Only Ablation

| Factor | Original |sel| | Ablated |sel| | Reduction |
|--------|-------------|-------------|-----------|
| Reward | 0.347 | 0.128 | **63.0%** |
| Probability | 0.265 | 0.571 | -115.7% |
| Salience | 0.252 | 0.042 | **83.2%** |
| Identity | 0.313 | 0.154 | **50.8%** |

**Interpretation:**
- Reward selectivity strongly depends on recurrence (56-63% reduction)
- Salience selectivity depends on I->E connections specifically (83% reduction)
- Some factors show paradoxical increases after ablation, suggesting complex interactions

## Implementation Details

### Model Changes (`src/model.py`)

1. **InputEmbedding class** (lines 17-90)
   - Attention embedding: Self-attention over 14 input channels
   - Linear embedding: Simple expansion for large input dimensions
   - Deep embedding: 2-layer MLP

2. **EIRNN updates** (lines 111-178)
   - New params: `input_embed_dim`, `input_embed_type`, `attention_heads`, `learnable_h0`, `h0_init`
   - Forward pass applies embedding before processing
   - Initial state uses learned h0 if available

### Training Configuration

```python
{
    'max_epochs': 2000,
    'patience': 300,
    'input_embed_dim': 56,
    'input_embed_type': 'attention',  # E0 only
    'attention_heads': 4,
    'learnable_h0': True,
    'h0_init': 0.1,
}
```

### Smart Embedding Selection

- E0 (14 inputs): Attention embedding (14 -> 56)
- E1/E2 (252 inputs): Linear embedding (252 -> 56)

Rationale: Attention over 252 tokens is computationally expensive and may not be meaningful for neuron-specific inputs.

## Gap Analysis

### Why Did Improvements Fall Short?

Expected: 0.35-0.40 (based on historical best)
Achieved: 0.19-0.22

Possible reasons:

1. **Different data characteristics**: The data may have different properties than the session used for Phase 3-5 experiments
   - Current: 42 neurons (14 E, 28 I)
   - Historical: 54 neurons (41 E, 13 I)

2. **Linear vs attention embedding**: E1/E2 used linear embedding due to high input dimensionality, missing the attention benefit

3. **Model capacity**: The E:I ratio enforcement creates a large model (140 units) that may be harder to train

4. **Fundamental limit**: The restructured inputs may have an inherent ceiling on achievable correlation

## Conclusions

### Successes

1. **Improvements work**: 18-36% increase in validation correlation
2. **Input restructuring preserved**: E2 >= E0 (110.7% ratio)
3. **Emergent selectivity enhanced**: Higher fractions than original experiments
4. **Recurrence dependence confirmed**: 56% reduction for reward with ablation

### Limitations

1. **Target not reached**: 0.19-0.22 vs 0.35-0.40 target
2. **Different from historical best**: May need investigation of data/model differences

## Recommendations

1. **Investigate data differences**: Compare current data (42 neurons) to historical session (54 neurons)

2. **Try attention for E1/E2**: Implement efficient attention for large input spaces (e.g., grouped attention)

3. **Hyperparameter tuning**: Grid search over embedding dimension, attention heads

4. **Consider ensemble**: Average predictions from multiple seeds/configurations

## Files Generated

```
results/fit_quality_improvement/
├── E0_seed0/
│   ├── model_best.pt
│   ├── W_rec.npy, W_in.npy, h0.npy
│   ├── history.json
│   ├── results.json
│   └── emergent_selectivity.json
├── E1_seed0/
│   └── [same structure]
├── E2_seed0/
│   ├── [same structure]
│   └── ablation_results.json
├── all_results.json
└── REPORT.md
```

## Code for Loading Results

```python
import json
import numpy as np

# Load results
with open('results/fit_quality_improvement/all_results.json') as f:
    results = json.load(f)

# Access metrics
print(f"E0: {results['E0_seed0']['best_val_correlation']:.4f}")
print(f"E2: {results['E2_seed0']['best_val_correlation']:.4f}")

# Load weights
W_rec = np.load('results/fit_quality_improvement/E2_seed0/W_rec.npy')
h0 = np.load('results/fit_quality_improvement/E2_seed0/h0.npy')
```
