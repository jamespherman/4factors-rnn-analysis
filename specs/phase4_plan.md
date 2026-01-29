# Phase 4 Experiment Plan
## January 24, 2026

## Overview

Phase 4 consists of targeted final improvements building on the Phase 3 best model (attention embedding, val corr = 0.3843) and a comprehensive diagnostic analysis to understand remaining error sources.

## Current Best Configuration (Baseline)

From Phase 3, the best configuration uses:
- **Attention-based input embedding** (14 inputs -> 56 dim with 4 heads)
- Gradient-balanced Sinkhorn optimal transport for L_trial
- ltrial_scale = 0.5
- ReduceLROnPlateau scheduler
- max_epochs = 500, patience = 100

**Best Validation Correlation: 0.3843**

---

## Phase 4 Experiments

### Experiment A: Learnable Initial State

**Hypothesis:** The RNN starts from x=0, but real neurons have non-zero baseline firing. A learnable initial hidden state may help match the first few time bins better.

**Implementation:**
- Add a learnable parameter `h0` of shape [n_total]
- Initialize to small positive values (0.1)
- Expand to batch dimension at forward pass start
- Options:
  - Per-neuron initial state (most flexible)
  - Per-population initial state (separate h0_E and h0_I)

**Expected Impact:**
- Should reduce error at trial onset
- May improve overall correlation if early bins are currently driving error

**Configuration:**
```python
{
    'name': 'attention_learnable_h0',
    'input_embed_type': 'attention',
    'input_embed_dim': 56,
    'attention_heads': 4,
    'learnable_h0': True,
    'h0_per_neuron': True,  # vs per-population
    'use_grad_balancing': True,
    'ltrial_scale': 0.5,
}
```

---

### Experiment B: Attention + AdamW

**Hypothesis:** Phase 3 showed AdamW with weight decay (0.3748) was #2 behind attention (0.3843). Combining attention embedding with AdamW regularization on embedding parameters may reduce overfitting.

**Implementation:**
- Use attention embedding (Phase 3 best)
- Apply AdamW with weight_decay=0.01 on embedding parameters only
- Other parameters use weight_decay=0

**Expected Impact:**
- May provide small improvement through better regularization
- Risk: Could interfere with attention's learned representations

**Configuration:**
```python
{
    'name': 'attention_adamw',
    'input_embed_type': 'attention',
    'input_embed_dim': 56,
    'attention_heads': 4,
    'optimizer': 'adamw',
    'weight_decay': 0.01,
    'use_grad_balancing': True,
    'ltrial_scale': 0.5,
}
```

---

### Experiment C: Attention + Learnable Initial State

**Hypothesis:** If both A (learnable h0) and B (attention+AdamW) provide improvements, combining attention with learnable initial state may stack benefits since they address different aspects (input representation vs. initial conditions).

**Implementation:**
- Combine attention embedding with learnable h0
- Only run if Experiment A shows improvement over baseline

**Configuration:**
```python
{
    'name': 'attention_h0',
    'input_embed_type': 'attention',
    'input_embed_dim': 56,
    'attention_heads': 4,
    'learnable_h0': True,
    'h0_per_neuron': True,
    'use_grad_balancing': True,
    'ltrial_scale': 0.5,
}
```

---

### Experiment D (Optional): Attention + Learnable Alpha

**Hypothesis:** Phase 2c found that I neurons need faster dynamics (alpha=0.72) than E neurons (alpha=0.54). This could potentially combine with attention embedding since they address different aspects.

**Caveat:** Phase 2c showed combinations often hurt rather than helped. This is lower priority.

**Implementation:**
- Combine attention embedding with per-neuron learnable alpha
- Initialize with optimal values from Phase 2c (E: 0.54, I: 0.72)

**Configuration:**
```python
{
    'name': 'attention_alpha_neuron',
    'input_embed_type': 'attention',
    'input_embed_dim': 56,
    'attention_heads': 4,
    'learnable_alpha': 'neuron',
    'alpha_e_init': 0.54,
    'alpha_i_init': 0.72,
    'use_grad_balancing': True,
    'ltrial_scale': 0.5,
}
```

---

## Experiment Order and Decision Tree

```
1. Run Experiment A (attention + learnable h0)
   |
   ├── If A > baseline (0.3843): Run C (attention + h0 without AdamW, use result)
   |
   └── Otherwise: Continue

2. Run Experiment B (attention + AdamW)
   |
   ├── If B > baseline: Consider as potential improvement
   |
   └── Otherwise: Continue

3. If neither A nor B helped:
   └── Run Experiment D (attention + learnable alpha) as final attempt

4. Final model = best of {baseline, A, B, C, D}
```

---

## Success Criteria

| Metric | Current Best | Target | Stretch |
|--------|--------------|--------|---------|
| Val PSTH Correlation | 0.3843 | 0.40 | 0.42 |
| Gap to Target | 3.9% | 0% | - |

---

## Implementation Notes

### Adding Learnable Initial State

In `src/model.py`, modify EIRNN:

```python
def __init__(self, ..., learnable_h0: bool = False, h0_per_neuron: bool = True):
    ...
    if learnable_h0:
        if h0_per_neuron:
            self.h0 = nn.Parameter(torch.full((self.n_total,), 0.1))
        else:
            # Per-population
            self.h0_e = nn.Parameter(torch.tensor(0.1))
            self.h0_i = nn.Parameter(torch.tensor(0.1))
    else:
        self.h0 = None
```

In `forward()`:
```python
if self.h0 is not None:
    x = self.h0.expand(batch_size, -1)
else:
    x = torch.zeros(batch_size, self.n_total, device=device)
```

---

## Output Files

```
results/phase4/
├── attention_learnable_h0_result.json
├── attention_learnable_h0_model_best.pt
├── attention_adamw_result.json
├── attention_adamw_model_best.pt
├── attention_h0_result.json (if run)
├── attention_h0_model_best.pt (if run)
├── attention_alpha_neuron_result.json (if run)
├── attention_alpha_neuron_model_best.pt (if run)
├── summary.json
└── comparison.png
```

---

## Timeline

1. Implement learnable h0 in model.py
2. Run Experiments A, B (can run in parallel)
3. Decide on C, D based on results
4. Run diagnostic analysis on best model
5. Write summary and recommendations
