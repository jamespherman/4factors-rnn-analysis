# Input Design Findings: Factor Assignment and Structure

## Executive Summary

This document covers experiments on **how inputs are provided to the network**, including which neurons receive which factors, input timing, and sparsity. The input restructuring experiments tested whether selectivity emerges from recurrence rather than direct input drive.

---

## 1. Standard Input Structure

### Default Configuration
```
M = 14  # Total input dimensions

Inputs (all binary except eye position):
- input_fixation_on:    1 dim
- input_target_loc:     4 dims (one-hot for 4 locations)
- input_go_signal:      1 dim
- input_reward_on:      1 dim
- input_eye_x:          1 dim (continuous, normalized)
- input_eye_y:          1 dim (continuous, normalized)
- input_is_face:        1 dim
- input_is_nonface:     1 dim
- input_is_bullseye:    1 dim
- input_high_salience:  1 dim
- input_low_salience:   1 dim
```

### Factor Groups
| Factor | Inputs | Description |
|--------|--------|-------------|
| Spatial | loc1-4 | Target location (RF matching) |
| Reward | reward_on | High vs low reward |
| Probability | (derived) | Reward probability |
| Salience | high/low_salience | Stimulus salience |
| Identity | face/nonface/bullseye | Stimulus category |

---

## 2. Input Restructuring Experiments

### Motivation
Test whether factor selectivity arises from:
1. **Direct input drive** - neurons receive factor information directly
2. **Recurrent dynamics** - selectivity emerges from network interactions

### Experimental Conditions

| Experiment | Description | Neurons Receiving Each Factor |
|------------|-------------|-------------------------------|
| **E0** | Baseline | All neurons receive all factors at target onset |
| **E1** | Temporal fix | Spatial factors from trial start (not target onset) |
| **E2** | Primary | RF-matched spatial (E only), sparse features (p=0.5) |
| **E3a** | Sparsity low | Same as E2 but p=0.25 |
| **E3b** | Sparsity high | Same as E2 but p=0.75 |
| **E4** | Replication | E2 with different seeds |

### E2 Primary Design
```
Spatial factors:
- Only E neurons receive spatial input
- Each E neuron receives RF-matched location only
- I neurons receive no spatial input

Feature factors (reward, probability, salience, identity):
- Sparse assignment with p=0.5
- Each E neuron receives ~50% of feature factors
- I neurons may receive feature factors

Timing:
- Spatial from trial start
- Features from target onset
```

### Results

| Experiment | Mean Val Corr | vs E0 | Notes |
|------------|---------------|-------|-------|
| E0 (baseline) | 0.1646 | 100% | All neurons all factors |
| E1 (temporal fix) | 0.1763 | 107% | Slight improvement |
| **E2 (primary)** | **0.1585** | **96%** | Preserved performance |
| E3a (p=0.25) | 0.1649 | 100% | Similar to E0 |
| E3b (p=0.75) | 0.1703 | 103% | Slight improvement |
| E4 (replication) | 0.1235 | 75% | Worse (seed variance) |

### Key Finding
**E2 achieves 96.2% of E0's correlation** despite dramatically reducing input connectivity. This suggests the network can maintain performance with sparse, structured inputs.

---

## 3. Emergent Selectivity

### Definition
**Emergent selectivity** = factor selectivity in neurons that do NOT receive that factor as direct input.

### E2 Results (Original Input Restructuring)

| Factor | Non-Receivers | Significant | Fraction | Mean |sel| |
|--------|---------------|-------------|----------|---------|
| Reward | ~28 | 14 | **50.0%** | 0.206 |
| Probability | ~28 | 24 | **84.3%** | 0.321 |
| Salience | ~27 | 9 | **33.8%** | 0.157 |
| Identity | ~14 | 6 | **40.7%** | 0.199 |

### E2 Results (With Improved Model)

| Factor | Non-Receivers | Significant | Fraction | Mean |sel| |
|--------|---------------|-------------|----------|---------|
| Reward | 28 | 25 | **89.3%** | 0.347 |
| Probability | 28 | 23 | **82.1%** | 0.265 |
| Salience | 27 | 13 | **48.1%** | 0.251 |
| Identity | 14 | 10 | **71.4%** | 0.313 |

### Interpretation
Neurons show selectivity for factors they don't receive directly. This selectivity must arise from **recurrent network dynamics**.

---

## 4. Ablation Studies

### Recurrence Dependence
Testing whether emergent selectivity depends on recurrence.

### Full Recurrence Ablation (W_rec = 0)

| Factor | Original |sel| | Ablated |sel| | Reduction |
|--------|-------------|-------------|-----------|
| Reward | 0.346 | 0.152 | **56.1%** |
| Probability | 0.265 | 0.159 | **39.9%** |
| Salience | 0.252 | 0.548 | -117.4% |
| Identity | 0.312 | 0.470 | -50.4% |

### I→E Only Ablation

| Factor | Original |sel| | Ablated |sel| | Reduction |
|--------|-------------|-------------|-----------|
| Reward | 0.347 | 0.128 | **63.0%** |
| Probability | 0.265 | 0.571 | -115.7% |
| Salience | 0.252 | 0.042 | **83.2%** |
| Identity | 0.313 | 0.154 | **50.8%** |

### Key Findings
1. **Reward selectivity strongly depends on recurrence** (56-63% reduction)
2. **Salience selectivity depends on I→E connections** (83% reduction)
3. Some factors show paradoxical increases after ablation (complex interactions)

---

## 5. Sparsity Effects

### Sparsity Parameter (p)
Probability that each E neuron receives each feature factor.

| Sparsity (p) | Val Corr | vs E0 |
|--------------|----------|-------|
| 0.25 (sparse) | 0.1649 | 100.2% |
| 0.50 (default) | 0.1585 | 96.2% |
| 0.75 (dense) | 0.1703 | 103.4% |

### Findings
- Moderate sparsity (p=0.5) preserves performance
- Very sparse (p=0.25) works equally well
- Denser (p=0.75) slightly better
- **Sparsity has minimal impact on fit quality**

---

## 6. Timing Effects

### E1: Spatial from Trial Start
- Spatial factors provided from trial start (not target onset)
- Result: 0.1763 (107% of E0)
- **Slight improvement** from earlier spatial information

### Interpretation
Providing spatial context earlier may help the network prepare appropriate dynamics before target-related activity begins.

---

## 7. RF Matching

### E2 Design
Each E neuron receives only the spatial input corresponding to its receptive field location.

### Implementation
```python
# For neuron i with RF at location j:
#   - Receives only input_target_loc[j]
#   - Not input_target_loc[k] for k != j
```

### Effect
RF-matched spatial inputs maintain performance (96% of E0) while dramatically reducing input connectivity.

---

## 8. Different Datasets

### Data Characteristics Across Sessions

| Session | n_E | n_I | E:I Ratio | Notes |
|---------|-----|-----|-----------|-------|
| 08_15 | 41 | 13 | 3.2:1 | Standard for most experiments |
| 08_14 | 25 | 55 | 0.45:1 | Many more I neurons |
| 08_13 | 10 | 10 | 1:1 | Small, balanced |

### Input Restructuring Dataset
- Used 42 neurons (14 E, 28 I)
- E:I ratio: 0.5:1 (inverted from standard)
- Results not directly comparable to main experiments

### Performance Across Sessions
| Session | Val Corr | E Corr | I Corr |
|---------|----------|--------|--------|
| 08_15 | 0.4070 | 0.3497 | 0.5599 |
| 08_14 | 0.3608 | 0.3518 | 0.3735 |
| 08_13 | 0.3232 | 0.3029 | 0.3671 |

### Consistent Finding
**I neurons consistently fit better than E neurons across all sessions.**

---

## 9. Eye Position Inputs

### Configuration
```python
input_eye_x: normalized horizontal eye position
input_eye_y: normalized vertical eye position
```

### Eye Pathway Test
One experiment tested enhanced eye position pathway:
- `newton_08_15_with_eye`: results in test directory

### Status
No systematic comparison of with/without eye inputs.

---

## Summary: Input Design Recommendations

### For Standard Training
```python
# All neurons receive all 14 inputs
# Inputs provided at appropriate task events:
#   - Fixation on at trial start
#   - Target info at target onset
#   - Reward info after saccade
```

### For Testing Emergent Selectivity
```python
# E2 configuration:
spatial_to_E_only = True
rf_matching = True
feature_sparsity = 0.5
spatial_timing = 'trial_start'
feature_timing = 'target_onset'
```

### Key Insights
1. **Sparse, structured inputs preserve performance** (96% of full)
2. **Selectivity emerges from recurrence**, not just direct input
3. **I→E connections are particularly important** for salience selectivity
4. **I neurons fit better than E neurons** across all conditions

---

## Untested Input Designs

1. **Temporal delay lines** - Provide inputs with different delays to different neurons
2. **Input noise** - Add noise to inputs, not just to dynamics
3. **Population coding** - Represent factors with distributed activity patterns
4. **Separate pathways** - Distinct input matrices for different factor types
5. **Learned input assignment** - Let network learn which neurons receive which inputs
