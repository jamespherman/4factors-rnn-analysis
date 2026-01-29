# Phase 3 Improvement Plan
## January 23, 2026

**Status: IMPLEMENTED** - Code changes complete, ready for execution.

This document outlines the implementation plan for Phase 3 experiments to improve the E-I RNN model beyond Phase 2c best (0.3762 validation PSTH correlation), targeting 0.40+.

---

## Background

Phase 2c achieved a breakthrough with **input embedding expansion** (Cover's theorem):
- Best result: 0.3762 validation correlation using 14 -> 56 learnable embedding
- This exceeded the original target (0.3614) by 4.1%

Phase 3 focuses on:
1. Further exploring the embedding space (larger dimensions, deeper networks)
2. Carefully testing embedding + single improvement combinations
3. Novel architecture variants (typed, recurrent, attention embeddings)

---

## Experiments to Run

### Phase 3a: Embedding Exploration (High Priority)

| # | Name | Description | Rationale |
|---|------|-------------|-----------|
| 1 | embed_112_learnable | 8x expansion (14 -> 112) | If 4x helped, 8x may help more |
| 2 | embed_168_learnable | 12x expansion (14 -> 168) | Test limits of expansion |
| 3 | deep_embed_56 | 2-layer MLP (14 -> 28 -> 56) | Deeper may learn better features |
| 4 | embed_56_higher_noise | 56D + noise_scale=0.15 | Phase 2c showed 0.15 > 0.10 |

### Phase 3b: Embedding + Single Improvement (Medium Priority)

| # | Name | Description | Rationale |
|---|------|-------------|-----------|
| 5 | embed_56_alpha_neuron_optimal | 56D + learnable alpha (E:0.54, I:0.72 init) | Embedding is preprocessing, alpha is dynamics - may be orthogonal |
| 6 | embed_56_lower_lr | 56D + lr=0.5e-3 | Slower learning may find better minimum |
| 7 | embed_56_adamw | 56D + AdamW (weight_decay=0.01 on embed) | Regularize embedding to prevent overfitting |

### Phase 3c: Architecture Variants (Lower Priority)

| # | Name | Description | Rationale |
|---|------|-------------|-----------|
| 8 | typed_embedding | Separate embeddings per input group | Different input types may need different processing |
| 9 | recurrent_embedding | GRU preprocessing | Capture temporal dependencies in inputs |
| 10 | attention_embedding | Self-attention on inputs | Learn which inputs matter when |

---

## Implementation Details

### 1. New Embedding Types in `src/model.py`

Extend `InputEmbedding` class to support:

```python
# Deep embedding (multi-layer MLP)
embed_type='deep'
embed_hidden_dim=28  # intermediate dimension
embed_n_layers=2     # number of layers

# Typed embedding (per-group)
embed_type='typed'
input_groups=[(0,1), (1,5), (5,7), (7,9), (9,14)]  # 5 groups
group_embed_dims=[8, 16, 8, 8, 16]  # dims per group (total=56)

# Recurrent embedding (GRU)
embed_type='recurrent'
gru_hidden_dim=32

# Attention embedding
embed_type='attention'
attention_heads=4
```

### 2. Input Groups for Typed Embedding

Based on the 14-dimensional input structure:

| Group | Indices | Content | Suggested Embed Dim |
|-------|---------|---------|---------------------|
| 1 | 0 | fixation_on | 8 |
| 2 | 1-4 | target_loc (one-hot) | 16 |
| 3 | 5-6 | go_signal, reward_on | 8 |
| 4 | 7-8 | eye_x, eye_y | 8 |
| 5 | 9-13 | target features (face, salience, etc.) | 16 |
| **Total** | 0-13 | 14 inputs | **56** |

### 3. AdamW with Parameter Groups

For `embed_56_adamw`, use separate weight decay:

```python
optimizer = torch.optim.AdamW([
    {'params': model.input_embed.parameters(), 'weight_decay': 0.01},
    {'params': other_params, 'weight_decay': 0.0}
], lr=1e-3)
```

---

## Configuration Details

All experiments use these base settings (from Phase 2c best):

```python
{
    'loss_type': 'correlation',
    'scheduler': 'plateau',
    'use_grad_balancing': True,
    'ltrial_scale': 0.5,
    'max_epochs': 500,
    'patience': 100
}
```

---

## Risk Assessment

| Experiment | Risk | Mitigation |
|------------|------|------------|
| embed_112/168 | Overfitting | Monitor train/val gap; add dropout if needed |
| deep_embed | Optimization difficulty | Add skip connections if fails |
| embed_56_alpha | May hurt (like Phase 2c combos) | Accept if fails; embedding alone is good |
| typed_embedding | Implementation complexity | Verify output dims carefully |
| recurrent_embedding | Training instability | Reduce GRU hidden dim if unstable |
| attention_embedding | Memory/compute | Reduce heads if needed |

---

## Success Criteria

| Metric | Phase 2c Best | Phase 3 Target | Stretch Goal |
|--------|---------------|----------------|--------------|
| Best Val Correlation | 0.3762 | 0.40 | 0.42 |
| Improvement | baseline | +6.3% | +11.8% |

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `src/model.py` | Add new embedding types to InputEmbedding class |
| `scripts/experiment_phase3.py` | New experiment script |
| `specs/phase3_experiment_results.md` | Results documentation (after completion) |

---

## Execution

```bash
python scripts/experiment_phase3.py \
    --data data/rnn_export_Newton_08_15_2025_SC.mat \
    --output results/phase3/ \
    --device cpu
```

Expected runtime: ~4-6 hours for all 10 experiments (500 epochs each with early stopping).
