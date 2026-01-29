# Phase 4 Summary: Learnable Initial State Optimization

## Objective

Improve validation PSTH correlation from Phase 3 best (0.3843) to target (0.40) by testing:
- A. Learnable initial hidden state (h0)
- B. AdamW optimizer with weight decay
- C. Combined h0 + AdamW
- D. Learnable per-neuron alpha (integration constant)

## Results

### Experiment Results Table

| Experiment | Config | Best Val Corr | vs Phase 3 | vs Target |
|------------|--------|---------------|-----------|-----------|
| **attention_learnable_h0** | h0=learnable | **0.4021** | **+4.6%** | **ACHIEVED** |
| attention_h0_adamw | h0 + AdamW | 0.3987 | +3.7% | -0.3% |
| attention_baseline | Phase 3 best | 0.3907 | +1.7% | -2.3% |
| attention_adamw | AdamW only | 0.3878 | +0.9% | -3.1% |
| attention_alpha_neuron | learnable alpha | 0.3789 | -1.4% | -5.3% |

### Key Findings

1. **Learnable h0 exceeded the 0.40 target** with val_corr = 0.4021, a 4.6% improvement over Phase 3

2. **Combining h0 with AdamW hurt performance** (0.3987 vs 0.4021) - weight decay may interfere with h0 learning

3. **AdamW alone provided modest benefit** (+0.9%) - regularization helps but is not transformative

4. **Learnable alpha hurt performance** (-1.4%) - per-neuron time constants add complexity without benefit

5. **Baseline improved from Phase 3** (0.3907 vs 0.3843) - possibly due to longer training (500 vs 200 epochs)

## Best Configuration

```python
{
    "name": "attention_learnable_h0",
    "input_embed_dim": 56,
    "input_embed_type": "attention",
    "attention_heads": 4,
    "learnable_h0": True,
    "h0_init": 0.1,
    "use_grad_balancing": True,
    "ltrial_scale": 0.5,
    "max_epochs": 500,
    "patience": 100
}
```

### Learned h0 Statistics

| Metric | Value |
|--------|-------|
| Mean | 0.291 |
| Std | 0.204 |
| Min | -0.422 |
| Max | 0.603 |

The model learned mostly positive initial states (92.3%), enabling non-zero baseline firing rates.

## Diagnostic Analysis Highlights

### Per-Neuron Performance
- Mean correlation: 0.317
- E neurons: 0.370 (better)
- I neurons: 0.150 (much worse)
- High variance: best neuron r=0.83, worst r=-0.31

### Temporal Error
- Error increases over time (early MSE=127.5, late MSE=140.8)
- Model tends to underpredict firing rates later in trials

### Population Structure (PCA)
- PC1 correlation: 0.750 (excellent)
- PC2 correlation: 0.335 (moderate)
- PC3 correlation: 0.380 (moderate)
- Model captures main population axis but loses fidelity on secondary modes

### Trial Variability
- Real Fano factor: 48.6
- Model Fano factor: 2.9
- Model severely underestimates trial-to-trial variability (17x lower)

## Recommendations

### For Production Use
The **attention_learnable_h0** configuration is ready for use:
- Val corr = 0.4021 exceeds the 0.40 target
- Stable training, completed 500 epochs
- Meaningful learned parameters (h0 distribution)

### For Further Improvement

**High Priority:**
1. **Improve I neuron fitting** - Consider higher loss weight for I neurons
2. **Address late-trial error** - Error accumulates over time

**Medium Priority:**
3. **Increase variability** - Raise noise_scale from 0.1 to better match real Fano factors
4. **Longer training** - Model was still improving at 500 epochs

**Lower Priority:**
5. **E/I-specific h0 initialization** - I neurons learned higher h0 (0.39 vs 0.26)
6. **PCA-space loss** - To better capture secondary population modes

## Phase 4 Deliverables

### Experiment Results
- `results/phase4/attention_learnable_h0_result.json` (BEST)
- `results/phase4/attention_h0_adamw_result.json`
- `results/phase4/attention_baseline_result.json`
- `results/phase4/attention_adamw_result.json`
- `results/phase4/attention_alpha_neuron_result.json`

### Model Checkpoints
- `results/phase4/attention_learnable_h0_model_best.pt` (BEST)
- `results/phase4/attention_h0_adamw_model_best.pt`
- `results/phase4/attention_baseline_model_best.pt`
- `results/phase4/attention_adamw_model_best.pt`
- `results/phase4/attention_alpha_neuron_model_best.pt`

### Diagnostic Analysis
- `results/phase4/diagnostics/diagnostic_summary.json`
- `results/phase4/diagnostics/*.png` (9 figures)
- `specs/phase4_diagnostic_analysis.md`

### Documentation
- `specs/phase4_plan.md`
- `specs/phase4_summary.md` (this file)

## Conclusion

**Phase 4 achieved its goal.** The learnable initial state (h0) approach successfully pushed validation PSTH correlation above 0.40, reaching 0.4021. This represents a cumulative improvement of:
- +4.6% over Phase 3 best (0.3843)
- +17.2% over Phase 1 baseline (0.343)

The model now captures meaningful aspects of neural dynamics:
- Strong PC1 trajectory match (r=0.75)
- Learned non-trivial initial state distribution
- Per-neuron correlations up to 0.83

Remaining challenges for future work:
- I neuron fitting (mean r=0.15)
- Late-trial error accumulation
- Trial-to-trial variability (Fano factor mismatch)
