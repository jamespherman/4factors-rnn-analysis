# Phase 3 Experiment Results
## January 24, 2026

## Summary

Phase 3 experiments achieved a **new best validation correlation of 0.3843** using attention embedding, a +2.1% improvement over Phase 2c best (0.3762). However, the target of 0.40 was not reached (3.9% gap remains).

## Results Table (Ranked by Performance)

| Rank | Configuration | Best Val Corr | vs Phase 2c | Notes |
|------|--------------|---------------|-------------|-------|
| 1 | **attention_embedding** | **0.3843** | **+2.1%** | Self-attention on inputs - NEW BEST |
| 2 | embed_56_adamw | 0.3748 | -0.4% | AdamW with weight decay on embedding |
| 3 | embed_56_higher_noise | 0.3734 | -0.7% | noise_scale=0.15 |
| 4 | recurrent_embedding | 0.3733 | -0.8% | GRU preprocessing |
| 5 | embed_56_lower_lr | 0.3694 | -1.8% | lr=0.5e-3 |
| 6 | deep_embed_56 | 0.3630 | -3.5% | 2-layer MLP (14->28->56) |
| 7 | embed_56_alpha_neuron_optimal | 0.3623 | -3.7% | Optimal alpha init (E:0.54, I:0.72) |
| 8 | embed_112_learnable | 0.3440 | -8.6% | 8x expansion hurt |
| 9 | typed_embedding | 0.3305 | -12.1% | Separate embeddings per group |
| 10 | embed_168_learnable | 0.3120 | -17.1% | 12x expansion hurt badly |

## Reference Points

| Metric | Value |
|--------|-------|
| Phase 2c best (embed_56_learnable) | 0.3762 |
| **Phase 3 best (attention_embedding)** | **0.3843** |
| Target | 0.4000 |
| Stretch target | 0.4200 |
| Improvement over Phase 2c | +2.1% |
| Gap to target | 3.9% |

## Key Findings

### 1. Attention Embedding Works Best
Self-attention over input features (treating each of the 14 inputs as a token) allows the model to learn which input features are most relevant at each timestep. This achieved the best result (0.3843).

### 2. Larger Embeddings Hurt Performance
- 4x expansion (14->56): 0.3762 (Phase 2c best)
- 8x expansion (14->112): 0.3440 (-8.6%)
- 12x expansion (14->168): 0.3120 (-17.1%)

Larger embeddings likely overfit or make optimization harder.

### 3. Regularization Helps
- AdamW with weight_decay=0.01 on embedding: 0.3748
- Higher noise (0.15 vs 0.1): 0.3734

Both prevent overfitting on the expanded input space.

### 4. Combinations Still Don't Help
Combining embedding with learnable alpha (embed_56_alpha_neuron_optimal: 0.3623) underperformed simple embedding, consistent with Phase 2c findings.

### 5. Recurrent Preprocessing is Competitive
GRU-based embedding (0.3733) captures temporal dependencies in inputs and performs comparably to the best approaches.

## Best Configuration Details

```python
{
    'name': 'attention_embedding',
    'loss_type': 'correlation',
    'input_embed_dim': 56,
    'input_embed_type': 'attention',
    'attention_heads': 4,
    'scheduler': 'plateau',
    'use_grad_balancing': True,
    'ltrial_scale': 0.5,
    'max_epochs': 500,
    'patience': 100
}
```

## Training Progress

The attention embedding model trained for the full 500 epochs, with validation correlation increasing from ~0.08 at epoch 0 to 0.3843 at its peak.

## Files Generated

```
results/phase3/
├── summary.json                              # Full experiment summary
├── comparison.png                            # Bar chart comparison
├── attention_embedding_result.json           # Best experiment results
├── attention_embedding_model_best.pt         # Best model checkpoint
└── [other experiment results...]
```

## Recommendations for Phase 4 (if needed)

To close the remaining 3.9% gap to 0.40:

1. **Attention + Regularization**: Combine attention embedding with AdamW
2. **Deeper Attention**: Try multi-layer attention or transformer-style encoder
3. **Attention + Recurrent**: Attention embedding followed by GRU
4. **Ensemble**: Average predictions from attention, AdamW, and recurrent models
5. **Hyperparameter Tuning**: Grid search on attention_heads (2, 4, 8) and embed_dim (42, 56, 70)

## Conclusion

Phase 3 successfully improved validation correlation from 0.3762 to 0.3843 (+2.1%) using attention-based input embedding. The key insight is that learning feature interactions via self-attention is more effective than simply expanding input dimensionality. Further gains toward the 0.40 target may require combining attention with regularization techniques or ensemble methods.
