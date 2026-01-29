# Current Best Configuration for E-I RNN Training

## Summary

This document summarizes the results of systematic experiments to optimize the L_trial (single-trial variability) loss for the E-I RNN model trained on SC neural data.

**Best Result**: Validation PSTH correlation of **0.3614** using gradient-balanced training with Sinkhorn optimal transport.

## Best Configuration

```python
# Training settings
gradient_balancing = True      # Normalize gradients from each loss
ltrial_scale = 0.5             # Scale L_trial gradient contribution
sinkhorn_epsilon = 0.1         # Entropy regularization for optimal transport
sinkhorn_iterations = 20       # Iterations for Sinkhorn-Knopp algorithm

# Standard hyperparameters
learning_rate = 1e-3
n_epochs = 500
patience = 100                 # Early stopping patience
```

## Experiment Results

21 configurations were tested across two experiment rounds. Results ranked by validation PSTH correlation:

| Rank | Configuration | Val Corr | Method | Sinkhorn ε | Scale |
|------|---------------|----------|--------|------------|-------|
| 1 | gradbal_scale_0.5 | **0.3614** | GradBal | 0.1 | 0.5x |
| 2 | gradbal_eps_0.05 | 0.3559 | GradBal | 0.05 | 1.0x |
| 3 | gradbal_scale_0.25 | 0.3550 | GradBal | 0.1 | 0.25x |
| 4 | gradbal_eps_0.2_scale_0.5 | 0.3468 | GradBal | 0.2 | 0.5x |
| 5 | gradbal_eps_0.1 | 0.3434 | GradBal | 0.1 | 1.0x |
| 6 | gradbal_eps_0.05_scale_0.5 | 0.3354 | GradBal | 0.05 | 0.5x |
| 7 | gradbal_eps_0.2 | 0.3316 | GradBal | 0.2 | 1.0x |
| 8 | **baseline_no_ltrial** | **0.3299** | None | - | - |
| 9 | gradbal_eps_1.0 | 0.3202 | GradBal | 1.0 | 1.0x |
| 10 | lambda_0.1_eps_0.1 | 0.3185 | Fixed λ | 0.1 | λ=0.1 |
| 11 | gradbal_warmup_100 | 0.3168 | GradBal | 0.1 | 1.0x |
| 12 | gradbal_eps_0.5 | 0.3040 | GradBal | 0.5 | 1.0x |
| 13 | gradbal_warmup_50 | 0.2949 | GradBal | 0.1 | 1.0x |
| 14 | gradbal_scale_2.0 | 0.2292 | GradBal | 0.1 | 2.0x |
| 15 | lambda_0.3_eps_1.0 | 0.2281 | Fixed λ | 1.0 | λ=0.3 |
| 16 | lambda_0.3_eps_0.1 | 0.1877 | Fixed λ | 0.1 | λ=0.3 |
| 17 | lambda_0.3_eps_0.5 | 0.1767 | Fixed λ | 0.5 | λ=0.3 |
| 18 | lambda_0.5_eps_0.1 | 0.1731 | Fixed λ | 0.1 | λ=0.5 |
| 19 | lambda_1.0_eps_0.1 | 0.1714 | Fixed λ | 0.1 | λ=1.0 |
| 20 | lambda_1.0_eps_0.5 | 0.1648 | Fixed λ | 0.5 | λ=1.0 |
| 21 | lambda_1.0_eps_1.0 | 0.1629 | Fixed λ | 1.0 | λ=1.0 |

## Key Findings

### What Worked

1. **Gradient Balancing is Essential**
   - All top 7 configurations use gradient balancing
   - Fixed lambda weighting causes gradient conflicts that hurt L_neuron optimization
   - Gradient balancing normalizes each loss's gradient to unit norm, preventing one loss from dominating

2. **Reduced L_trial Contribution (scale=0.5)**
   - Best result uses 0.5x scale for L_trial gradients
   - L_neuron (PSTH matching) should remain the primary objective
   - Too much L_trial emphasis (scale=2.0) severely hurts performance

3. **Tight Sinkhorn Epsilon (0.05-0.1)**
   - Lower epsilon produces harder (more deterministic) trial assignments
   - ε=0.1 works well; ε=0.05 also performs strongly
   - Higher epsilon (≥0.5) leads to overly soft matching that doesn't capture trial structure

4. **No Warmup Needed**
   - Adding L_trial from epoch 0 works fine
   - Warmup periods (50 or 100 epochs without L_trial) actually hurt final performance

### What Didn't Work

1. **Fixed Lambda Weighting**
   - All fixed-lambda configurations performed below baseline
   - λ≥0.3 causes catastrophic degradation (correlation drops to 0.16-0.19)
   - Gradient conflicts prevent proper optimization of L_neuron

2. **High L_trial Scale**
   - scale=2.0 dropped performance to 0.229 (below baseline)
   - L_trial should be a secondary objective, not primary

3. **Soft Sinkhorn Assignments**
   - ε≥0.5 produces too-soft probability matrices
   - Loses the bijective matching property needed for proper trial alignment

## Technical Details

### Gradient Balancing

From Défossez et al. (2023), gradient balancing computes a balanced loss:

```
L_balanced = L_neuron + scale * (||∇L_neuron|| / ||∇L_trial||) * L_trial
```

This ensures both losses contribute equally (modulated by scale) to parameter updates regardless of their absolute magnitudes.

### Sinkhorn Optimal Transport

The L_trial loss uses Sinkhorn-Knopp algorithm to find optimal trial-to-trial matching:

1. Compute pairwise distances between model and target single-trial responses
2. Apply Sinkhorn iterations to find doubly-stochastic assignment matrix
3. Compute weighted average distance using soft assignments

The entropy-regularized optimal transport objective:
```
min_P <P, D> - ε H(P)
subject to: P1 = 1/n, P^T1 = 1/m
```

Where D is the distance matrix, P is the transport plan, and H(P) is entropy.

## Improvement Over Baseline

| Metric | Baseline | Best Config | Improvement |
|--------|----------|-------------|-------------|
| Val PSTH Correlation | 0.3299 | 0.3614 | +9.5% |

The L_trial loss successfully improves model performance when properly implemented with gradient balancing and appropriate hyperparameters.

## Recommended Usage

To train with the optimal configuration:

```python
python scripts/train_model.py \
    --gradient_balancing \
    --ltrial_scale 0.5 \
    --sinkhorn_epsilon 0.1 \
    --n_epochs 500 \
    --patience 100
```

## Files

- Experiment scripts: `scripts/experiment_ltrial.py`, `scripts/experiment_ltrial_v2.py`
- Results: `results/ltrial_experiments/`, `results/ltrial_experiments_v2/`
- Loss implementation: `src/losses.py` (contains `sinkhorn_assignment()` and `compute_L_trial()`)
