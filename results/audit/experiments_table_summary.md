# Experiments Summary Table

Generated: 2026-01-31

## Overview

**Total experiments catalogued: 98**

| Phase | Count | Date Range | Focus |
|-------|-------|------------|-------|
| L_trial v1 | 12 | Jan 2026 | Fixed lambda vs gradient balancing |
| L_trial v2 | 9 | Jan 2026 | Scale and epsilon tuning |
| Phase 2 | 17 | Jan 23, 2026 | Learnable alpha/tau, low-rank |
| Phase 2c | 16 | Jan 23, 2026 | Combinations, input embedding |
| Phase 3 | 10 | Jan 24, 2026 | Attention, deeper architectures |
| Phase 4 | 5 | Jan 24, 2026 | Learnable h0 |
| Phase 5 | 8 | Jan 24, 2026 | Limitation analysis |
| Phase 6 | 3 | Jan 25, 2026 | Cross-session replication |
| Final | 1 | Jan 25, 2026 | Production model |
| Input Restructuring | 6 | Jan 29, 2026 | RF-matched sparse inputs |
| Fit Quality Improvement | 3 | Jan 30, 2026 | Combined improvements |

---

## Top 10 Experiments (by Validation PSTH Correlation)

| Rank | Experiment | Phase | Val Corr | Key Features |
|------|------------|-------|----------|--------------|
| 1 | **extended_training** | phase5 | **0.4121** | 1000 epochs, attention+h0 |
| 2 | **final_model** | final | **0.4070** | 2000 epochs, best production config |
| 3 | **replication_08_15** | phase6 | **0.4070** | Same as final_model |
| 4 | **attention_learnable_h0** | phase4 | **0.4021** | Target achieved! |
| 5 | low_rank_50_phase5 | phase5 | 0.3990 | Low-rank with best config |
| 6 | attention_h0_adamw | phase4 | 0.3987 | h0 + AdamW combination |
| 7 | attention_baseline | phase4 | 0.3907 | Phase 3 config re-run |
| 8 | learnable_tau_phase5 | phase5 | 0.3880 | Learnable tau (didn't help) |
| 9 | attention_adamw | phase4 | 0.3878 | AdamW alone |
| 10 | noise_0.3_phase5 | phase5 | 0.3850 | Higher noise test |

---

## Bottom 10 Experiments (Excluding Failed)

| Rank | Experiment | Phase | Val Corr | Reason for Poor Performance |
|------|------------|-------|----------|---------------------------|
| 1 | E4_input_restructuring | input_restructuring | 0.1235 | Different seed replication |
| 2 | E2_input_restructuring | input_restructuring | 0.1585 | Sparse RF-matched inputs |
| 3 | lambda_1.0_eps_1.0 | ltrial_v1 | 0.1629 | Both lambda and epsilon bad |
| 4 | lambda_1.0_eps_0.5 | ltrial_v1 | 0.1648 | Strong lambda + soft sinkhorn |
| 5 | E0_input_restructuring | input_restructuring | 0.1646 | Baseline for restructuring |
| 6 | lambda_1.0_eps_0.1 | ltrial_v1 | 0.1714 | Fixed lambda too strong |
| 7 | lambda_0.5_eps_0.1 | ltrial_v1 | 0.1731 | Fixed lambda too strong |
| 8 | lambda_0.3_eps_0.5 | ltrial_v1 | 0.1767 | Soft sinkhorn hurt |
| 9 | E1_input_restructuring | input_restructuring | 0.1763 | Temporal fix experiment |
| 10 | lambda_0.3_eps_0.1 | ltrial_v1 | 0.1877 | Fixed lambda too strong |

---

## Failed Experiments

| Experiment | Phase | Val Corr | Failure Mode |
|------------|-------|----------|--------------|
| learnable_tau_neuron_fixed | phase2 | 0.0234 | alpha > 1 due to tau < dt |
| learnable_tau_population_fixed | phase2 | 0.0205 | alpha > 1 due to tau < dt |

**Root cause:** Initial TAU_MIN = 10ms was less than dt = 25ms, causing integration constant alpha = dt/tau > 1, which broke RNN dynamics. Fixed in Phase 2c by setting TAU_MIN >= dt.

---

## Key Configuration Patterns

### Best Configuration (Production)

```
learning_rate: 0.001
max_epochs: 2000
patience: 300
tau: 50.0
noise_scale: 0.1
input_embed_dim: 56
input_embed_type: attention
attention_heads: 4
learnable_h0: true
h0_init: 0.1
use_grad_balancing: true
ltrial_scale: 0.5
sinkhorn_epsilon: 0.1
scheduler: plateau
```

### Configuration That Doesn't Help

- Learnable alpha (adds complexity, no benefit with attention)
- Larger embeddings (112, 168 hurt)
- Faster tau (25ms hurt)
- Higher noise (>0.1 hurt)
- Bypassing Dale's law (hurt)
- Warmup for L_trial (hurt)
- Fixed lambda weighting (hurt)
- Soft Sinkhorn (epsilon >= 0.5 hurt)
- Combining multiple improvements (usually hurt)

---

## Improvement Timeline

| Date | Milestone | Val Corr | Improvement |
|------|-----------|----------|-------------|
| Jan 23 | Phase 2 baseline | 0.2901 | - |
| Jan 23 | learnable_alpha_neuron | 0.3449 | +18.9% |
| Jan 23 | embed_56_learnable | 0.3762 | +29.7% |
| Jan 24 | attention_embedding | 0.3843 | +32.5% |
| Jan 24 | attention_learnable_h0 | 0.4021 | +38.6% (TARGET) |
| Jan 24 | extended_training | 0.4121 | +42.1% (BEST) |
| Jan 25 | final_model (2000 epochs) | 0.4070 | +40.3% (PRODUCTION) |

---

## Cross-Session Performance

| Session | Neurons (E/I) | Val Corr | Relative |
|---------|---------------|----------|----------|
| Newton_08_15 | 54 (41E/13I) | 0.4070 | 100% |
| Newton_08_14 | 80 (25E/55I) | 0.3608 | 89% |
| Newton_08_13 | 20 (10E/10I) | 0.3232 | 79% |

**Note:** Performance varies with neuron count and E/I ratio. All sessions show consistent I > E correlation (I neurons fit better).

---

## Dataset Notes

Most experiments used `Newton_08_15_2025_SC` with 54 neurons (41 E, 13 I).

Input restructuring and fit quality improvement experiments used a subset with 42 neurons (14 E, 28 I) due to different data filtering criteria.

Results between these datasets are not directly comparable.
