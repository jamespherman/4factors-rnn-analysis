# Phase 3 Improvement Recommendations
## January 23, 2026

This document provides recommendations for further improving the E-I RNN model beyond the Phase 2c best of **0.3762** validation PSTH correlation.

---

## Current Status

| Metric | Value |
|--------|-------|
| Phase 2c Best | 0.3762 |
| Target (prior work) | 0.3614 |
| Status | **Exceeded target by 4.1%** |

The breakthrough came from **input embedding expansion** (Cover's theorem) - expanding 14 inputs to 56 dimensions via a learnable linear layer + ReLU.

---

## Key Insights from Phase 2c

1. **Input embedding is highly effective** - Cover's theorem works for RNN fitting
2. **Combinations don't stack** - Individual improvements compete rather than complement
3. **Larger expansion is better** - 56D > 28D for embedding
4. **Learnable > fixed** - Learnable embeddings outperform random/time-lag
5. **Higher noise marginally helps** - 0.15 > 0.10 > 0.05

---

## Phase 3 Priority Experiments

### Tier 1: Expand on Input Embedding Success

Since input embedding was the breakthrough, explore this direction further:

#### 1.1 Larger Embedding Dimensions
```python
embedding_sweep = [
    {'input_embed_dim': 112, 'input_embed_type': 'learnable'},  # 8x
    {'input_embed_dim': 168, 'input_embed_type': 'learnable'},  # 12x
    {'input_embed_dim': 224, 'input_embed_type': 'learnable'},  # 16x
]
```
**Rationale:** If 56D (4x) helped significantly, larger expansions may help more. However, watch for overfitting.

#### 1.2 Multi-Layer Embedding
```python
class DeepInputEmbedding(nn.Module):
    def __init__(self, n_inputs, hidden_dim, output_dim):
        self.layers = nn.Sequential(
            nn.Linear(n_inputs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
```
**Rationale:** A deeper embedding network may learn more complex input transformations.

#### 1.3 Embedding + Best Other Improvement (Careful!)
Previous combinations hurt, but embedding is fundamentally different (preprocessing vs dynamics). Try:
```python
{
    'input_embed_dim': 56,
    'learnable_alpha': 'neuron',
    'alpha_e_init': 0.54,
    'alpha_i_init': 0.72,
}
```
**Rationale:** Embedding addresses input representation, alpha addresses dynamics - these may be orthogonal. Note that embed_28_alpha_neuron failed (0.3167), so be cautious.

---

### Tier 2: Architecture Modifications

#### 2.1 Separate Embedding for Different Input Types
The 14 inputs likely include different signal types (e.g., position, velocity, task cues). Create separate embeddings:
```python
class TypedInputEmbedding(nn.Module):
    def __init__(self, input_groups, embed_dims):
        # input_groups = [(0, 4), (4, 8), (8, 14)]  # position, velocity, task
        # embed_dims = [16, 16, 24]
        self.embeddings = nn.ModuleList([
            nn.Sequential(nn.Linear(end-start, dim), nn.ReLU())
            for (start, end), dim in zip(input_groups, embed_dims)
        ])
```
**Rationale:** Different input types may benefit from different embedding structures.

#### 2.2 Recurrent Embedding
Add recurrence to the embedding layer:
```python
class RecurrentEmbedding(nn.Module):
    def __init__(self, n_inputs, embed_dim):
        self.gru = nn.GRU(n_inputs, embed_dim, batch_first=True)
```
**Rationale:** Temporal dependencies in inputs may be better captured with a recurrent layer.

#### 2.3 Attention-Based Input Weighting
```python
class AttentionEmbedding(nn.Module):
    def __init__(self, n_inputs, embed_dim):
        self.query = nn.Linear(n_inputs, embed_dim)
        self.key = nn.Linear(n_inputs, embed_dim)
        self.value = nn.Linear(n_inputs, embed_dim)
```
**Rationale:** Allow the model to attend to relevant inputs at each timestep.

---

### Tier 3: Training Improvements

#### 3.1 Data Augmentation
```python
def augment_trial(inputs, targets, noise_std=0.01):
    """Add small perturbations to create synthetic trials."""
    augmented_inputs = inputs + torch.randn_like(inputs) * noise_std
    return augmented_inputs, targets
```
**Rationale:** More training data through augmentation may improve generalization.

#### 3.2 Curriculum Learning for Embedding
Start with a smaller embedding dimension and gradually increase:
```python
def curriculum_embedding_dim(epoch, max_epochs, min_dim=14, max_dim=56):
    progress = min(1.0, epoch / (max_epochs * 0.5))
    return int(min_dim + (max_dim - min_dim) * progress)
```
**Rationale:** Easier optimization path may find better solutions.

#### 3.3 Different Optimizers
Try AdamW with weight decay specifically on embedding parameters:
```python
optimizer = torch.optim.AdamW([
    {'params': model.input_embed.parameters(), 'weight_decay': 0.01},
    {'params': other_params, 'weight_decay': 0.0}
], lr=1e-3)
```

---

### Tier 4: Loss Function Refinements

#### 4.1 Embedding-Specific Regularization
```python
def embedding_orthogonality_loss(embedding_layer):
    """Encourage diverse embedding features."""
    W = embedding_layer.linear.weight  # [embed_dim, n_inputs]
    WWT = W @ W.T
    identity = torch.eye(W.shape[0], device=W.device)
    return ((WWT - identity) ** 2).mean()
```
**Rationale:** Orthogonal embedding features may improve expressiveness.

#### 4.2 Contrastive Loss for Trial Matching
```python
def contrastive_L_trial(model_rates, target_rates, temperature=0.1):
    """Use contrastive learning for trial matching."""
    # Encourage each model trial to be close to its matched target
    # and far from other targets
```
**Rationale:** May provide stronger gradients for trial matching.

---

## Experiments NOT Recommended

Based on Phase 2c findings, avoid:

1. **Multiple combinations** - They consistently hurt performance
2. **Poisson L_trial with other changes** - Interferes with gradient balancing
3. **Low-rank + other improvements** - Regularization conflicts
4. **Very low noise (< 0.05)** - Reduces trial variability

---

## Recommended Experiment Order

**Phase 3a: Embedding Exploration (High Priority)**
1. embed_112_learnable (8x expansion)
2. embed_168_learnable (12x expansion)
3. deep_embed_56 (2-layer embedding)
4. embed_56_higher_noise (0.15)

**Phase 3b: Embedding + Single Improvement (Medium Priority)**
5. embed_56_alpha_neuron_careful (with optimal init)
6. embed_56_lower_lr (0.5e-3)
7. embed_56_adamw (with weight decay)

**Phase 3c: Architecture Variants (Lower Priority)**
8. typed_embedding (separate by input type)
9. recurrent_embedding (GRU pre-processing)
10. attention_embedding

---

## Success Criteria

| Metric | Current | Phase 3 Target |
|--------|---------|----------------|
| Val PSTH Correlation | 0.3762 | 0.40+ |
| Final Val Correlation | 0.3644 | 0.38+ |
| Training Stability | Good | Maintain |
| Biological Plausibility | Good | Maintain |

---

## Implementation Notes

### Adding Larger Embeddings
In `src/model.py`, the InputEmbedding class already supports arbitrary dimensions:
```python
model = create_model_from_data(
    ...,
    input_embed_dim=112,  # 8x expansion
    input_embed_type='learnable',
)
```

### Creating Deep Embeddings
Add a new class to `src/model.py`:
```python
class DeepInputEmbedding(nn.Module):
    def __init__(self, n_inputs, hidden_dim, output_dim, n_layers=2):
        super().__init__()
        layers = [nn.Linear(n_inputs, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.extend([nn.Linear(hidden_dim, output_dim), nn.ReLU()])
        self.network = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, x):
        return self.network(x)
```

### Experiment Script
Create `scripts/experiment_phase3.py` following the pattern of `experiment_phase2c.py`.

---

## Risk Assessment

| Approach | Potential Gain | Risk | Recommendation |
|----------|----------------|------|----------------|
| Larger embedding | High | Overfitting | Try with caution |
| Deep embedding | Medium | Optimization difficulty | Worth trying |
| Embed + alpha | Low-Medium | May hurt like other combos | Be cautious |
| Curriculum | Medium | Implementation complexity | Worth trying |
| Augmentation | Medium | May not help | Worth trying |

---

## Conclusion

The Phase 2c experiments identified **input embedding expansion** as a breakthrough technique that exceeded the target. Phase 3 should focus on:

1. **Further exploring the embedding space** (larger dimensions, deeper networks)
2. **Carefully testing embedding + single improvement** (may work unlike other combinations)
3. **Training refinements** (data augmentation, curriculum learning)

The path to 0.40+ validation correlation likely involves more sophisticated input preprocessing rather than additional changes to the RNN dynamics or loss function.
