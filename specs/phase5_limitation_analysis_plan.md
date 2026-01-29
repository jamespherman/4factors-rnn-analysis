# Phase 5: Limitation Analysis Plan

## Objective

Understand what fundamentally limits model performance at the 0.40 validation PSTH correlation level, before deciding whether to pursue further optimization or pivot to scientific analysis of learned weights.

## Current State

- **Best model**: attention_learnable_h0 (val_corr = 0.4021)
- **Key issues identified**:
  1. Model outputs too smooth (misses fast fluctuations)
  2. I neurons fit poorly (mean r=0.15 vs E neurons r=0.37)
  3. Trial-to-trial variability 17x too low (Fano factor: model=2.9, real=48.6)
  4. Training curves hadn't plateaued at 500 epochs

## Experiments

### Experiment 0: Extended Training (Background)
- **Config**: Best Phase 4 config with 1000 epochs, patience=150
- **Hypothesis**: Model was still improving; may gain incremental improvement
- **Expected outcome**: Modest improvement (+1-3%) if curve hadn't plateaued

### Experiment 1: Faster Dynamics (τ=25ms)
- **Config**: Fixed τ=25ms (vs default 50ms)
- **Hypothesis**: Smoothness caused by slow time constants
- **Expected outcomes**:
  - If improves: τ was bottleneck, consider learnable τ
  - If unchanged: Smoothness from other source (network dynamics, loss function)
  - If worse: Fast dynamics unstable or wrong for this data

### Experiment 2: Higher Noise (scale=0.3 and 0.5)
- **Config**: noise_scale=0.3 and noise_scale=0.5 (vs default 0.1)
- **Hypothesis**: Low Fano factor due to insufficient noise injection
- **Expected outcomes**:
  - If Fano factor increases without hurting correlation: noise was bottleneck
  - If correlation drops: noise interferes with PSTH matching
  - Need to find optimal noise level

### Experiment 3: Relaxed E/I Assignment
- **Config**: bypass_dale=True (unconstrained weights)
- **Hypothesis**: Forcing recorded E/I types to model E/I may be wrong
- **Analysis**:
  - Does performance improve?
  - Do learned weight signs correlate with original E/I labels?
  - What fraction of "classic" neurons have inhibitory-like connectivity?

### Experiment 4: Learnable τ (Revisited)
- **Config**: learnable_tau='population', τ_e_init=50, τ_i_init=35
- **Hypothesis**: With strong baseline (0.40), per-population τ may help
- **Expected outcomes**:
  - Learned τ_e and τ_i values
  - Whether I neurons prefer faster dynamics

### Experiment 5: Poisson L_trial (Revisited)
- **Config**: use_poisson_distance=True in L_trial
- **Hypothesis**: At lower baseline (0.29), Poisson helped (+16%)
- **Expected outcomes**:
  - Does it still help at 0.40 baseline?
  - Interaction with attention embedding

### Experiment 6: Low-Rank Connectivity
- **Config**: low_rank=50 (vs full rank)
- **Hypothesis**: Regularization may improve generalization
- **Expected outcomes**:
  - Trade-off between regularization and expressivity
  - Whether low-rank finds cleaner structure

## Analytical Deep-Dives

### I Neuron Analysis
1. **Input-output correlations**: What inputs most drive I neurons in real data?
   - Correlate each input channel with each I neuron's PSTH
   - Compare to model's I neuron input weights

2. **Missing inputs**: Are I neurons responsive to unmodeled signals?
   - Check residual structure for systematic patterns
   - Temporal autocorrelation of residuals

3. **Temporal dynamics**: Do I neurons have faster dynamics than E?
   - Autocorrelation analysis of real vs model PSTHs
   - Power spectra by neuron type

4. **Signal-to-noise**: Is poor fit due to low SNR or active errors?
   - Compare variance explained vs residual variance
   - Correlation with neuron mean firing rate

### Spectral Analysis
1. **Power spectra**: Where does model lose high-frequency content?
   - Compute power spectra of real and model PSTHs
   - Identify cutoff frequency

2. **Neuron-type specificity**: Is smoothness uniform?
   - Separate analysis for E vs I neurons
   - Identify if specific neurons are smoothest

## Expected Insights

| Question | Experiment | Positive Signal | Negative Signal |
|----------|------------|-----------------|-----------------|
| Is τ the bottleneck? | 1, 4 | τ=25ms or learnable helps | No change |
| Is noise the bottleneck? | 2 | Higher noise helps Fano | Correlation drops |
| Does E/I constraint hurt? | 3 | bypass_dale helps | No change |
| Are we at architecture limit? | 0, 4, 5, 6 | Combinations help | All plateau at ~0.40 |
| Why are I neurons hard? | Analysis | Clear driver identified | Noisy residuals |

## Success Criteria

1. **Identify primary bottleneck**: At least one experiment should show clear improvement
2. **Understand I neuron issue**: Analysis should explain why I neurons are harder
3. **Clear recommendation**: Path forward should be clear (optimize more or analyze weights)

## Deliverables

- `results/phase5/` - All experiment results and checkpoints
- `results/phase5/analysis/` - All analysis figures
- `specs/phase5_limitation_analysis_report.md` - Full analysis report
- Summary answering: best correlation, primary limitation, next steps
