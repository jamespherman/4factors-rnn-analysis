# Experiment Audit Plan

## Overview

This document outlines the comprehensive audit of all RNN fitting experiments conducted in this repository. The goal is to create a clear, unified view of what has been tried, what worked, what didn't work, and what remains to be explored.

## Scope

### Experiments Identified

Based on repository exploration, the following experiment groups have been identified:

| Phase | Experiments | Date | Primary Focus |
|-------|-------------|------|---------------|
| L_trial Experiments | 21 configs | Jan 2026 | Gradient balancing, Sinkhorn OT |
| Phase 2 | 17 configs | Jan 23, 2026 | Learnable alpha/tau, low-rank, regularization |
| Phase 2c | 16 configs | Jan 23, 2026 | Combinations, input embedding |
| Phase 3 | 10 configs | Jan 24, 2026 | Attention embedding, deeper architectures |
| Phase 4 | 5 configs | Jan 24, 2026 | Learnable h0, AdamW |
| Phase 5 | 8 configs | Jan 24, 2026 | Limitation analysis, extended training |
| Phase 6 | 3 sessions | Jan 25, 2026 | Cross-session replication |
| Final Model | 1 config | Jan 25, 2026 | Production model (2000 epochs) |
| Input Restructuring | 18 runs | Jan 29, 2026 | RF-matched sparse inputs |
| Fit Quality Improvement | 3 configs | Jan 30, 2026 | Attention + h0 with restructured inputs |
| Early Tests | ~15 configs | Various | Curriculum, large model, noise, etc. |

**Total estimated experiments: ~120+**

---

## Directories to Examine

### 1. Results Directory (`results/`)

| Directory | Contents | Information to Extract |
|-----------|----------|----------------------|
| `final_model/` | Production model | config.json, training_report.md, metrics |
| `phase2/` | 17 experiments | summary.json, *_result.json |
| `phase2c/` | 16 experiments | summary.json, *_result.json |
| `phase3/` | 10 experiments | summary.json, *_result.json |
| `phase4/` | 5 experiments | diagnostics/, *_result.json |
| `phase5/` | 8 experiments | summary.json, analysis/ |
| `replication/` | 3 sessions | per-session results, comparison/ |
| `input_restructuring/` | 18 runs | REPORT.md, per-experiment folders |
| `fit_quality_improvement/` | 3 configs | all_results.json, REPORT.md |
| `ltrial_experiments/` | 12 configs | summary.json, per-config results.json |
| `ltrial_experiments_v2/` | 9 configs | summary.json, per-config results.json |
| `improvements_test/` | 6 configs | *_result.json |
| `test_*` directories | ~10 experiments | config.json, history.json |
| `Newton_08_14_*` | 6 variants | config.json, history.json |
| `e_only_model/` | 1 experiment | model outputs |
| `newton_08_15_*` | 2 experiments | config.json, psth_summary.json |
| `ltrial_disabled/` | 1 experiment | results.json |
| `conditioned_loss_08_15/` | Evaluation | evaluation_summary.md |

### 2. Specs Directory (`specs/`)

| File | Information to Extract |
|------|----------------------|
| `current_best_config.md` | Best configuration reference |
| `improvement_experiment_results_01232026.md` | Phase 2 detailed results |
| `phase2c_experiment_results_01232026.md` | Phase 2c detailed results |
| `phase3_experiment_results.md` | Phase 3 detailed results |
| `phase4_summary.md` | Phase 4 detailed results |
| `phase5_limitation_analysis_report.md` | Phase 5 detailed results |
| `phase6_replication_report.md` | Replication findings |
| `ARCHITECTURE_SPEC.md` | Model architecture details |
| `TRAINING_SPEC.md` | Loss functions, optimization |
| `*_plan.md` files | Rationale for experiments |

### 3. Scripts Directory (`scripts/`)

| Script | Information to Extract |
|--------|----------------------|
| `train_final_model.py` | Production configuration |
| `experiment_phase*.py` | Experiment configurations |
| `experiment_*.py` | Various experiment configs |
| `run_input_restructuring.py` | Input restructuring setup |
| `generate_fit_quality_figures.py` | Analysis approach |

---

## Data Extraction Schema

For each experiment, extract the following fields:

### Configuration Fields

| Field | Description | Source |
|-------|-------------|--------|
| `experiment_id` | Unique identifier | Directory/file name |
| `phase` | Experiment phase | Directory structure |
| `date` | When run | Timestamp or file date |
| `dataset` | Data file used | config.json |
| `n_neurons` | Total neurons (E + I) | config.json or data |
| `hidden_size` | Total RNN units | Computed from E:I ratio |
| `learning_rate` | Initial LR | config.json |
| `max_epochs` | Training limit | config.json |
| `patience` | Early stopping patience | config.json |
| `scheduler` | LR scheduler type | config.json or script |
| `tau` | Time constant | config.json |
| `noise_scale` | Noise level | config.json |
| `input_embed_dim` | Embedding dimension | config.json |
| `input_embed_type` | Embedding type | config.json |
| `attention_heads` | Attention heads | config.json |
| `learnable_h0` | Whether h0 learned | config.json |
| `learnable_alpha` | Whether alpha learned | config.json |
| `learnable_tau` | Whether tau learned | config.json |
| `low_rank` | Low-rank constraint | config.json or script |
| `use_grad_balancing` | Gradient balancing | config.json |
| `ltrial_scale` | L_trial weight | config.json |
| `loss_type` | Loss function type | config.json or script |
| `seed` | Random seed | config.json |

### Result Fields

| Field | Description | Source |
|-------|-------------|--------|
| `best_val_corr` | Best validation PSTH correlation | results.json |
| `final_val_corr` | Final epoch correlation | results.json |
| `best_epoch` | Epoch of best result | results.json |
| `epochs_trained` | Total epochs run | history.json |
| `train_corr` | Training correlation | history.json |
| `e_neuron_corr` | E neuron mean correlation | metrics/ |
| `i_neuron_corr` | I neuron mean correlation | metrics/ |
| `converged` | Whether training converged | Inferred from epochs |
| `training_time` | Training duration | results.json |
| `notes` | Special observations | Manual review |

---

## Analysis to Perform

### 1. Ranking Analysis
- Rank all experiments by best_val_corr
- Identify top 10 configurations
- Identify worst 10 configurations

### 2. Parameter Impact Analysis
- Learning rate vs performance
- Hidden size vs performance
- Epochs vs performance
- Noise scale vs performance
- Embedding dimension vs performance

### 3. Technique Effectiveness

| Technique | Metric | Analysis |
|-----------|--------|----------|
| Attention embedding | Mean improvement | Compare with/without |
| Learnable h0 | Mean improvement | Compare with/without |
| Input embedding | Mean improvement | Compare dim 0 vs 28 vs 56 |
| Gradient balancing | Mean improvement | Compare with/without |
| Low-rank constraint | Mean improvement | Compare ranks |
| Learnable alpha/tau | Mean improvement | Compare variants |
| Extended training | Mean improvement | 500 vs 1000 vs 2000 epochs |

### 4. Combination Analysis
- Which combinations were tested?
- Which combinations showed synergy?
- Which combinations showed interference?

### 5. Session Comparison
- Performance across Newton sessions
- Which findings replicate?

---

## Output Files

### 1. `experiments_table.csv`

Master table with one row per experiment, all configuration and result columns.

### 2. `experiments_table_summary.md`

Markdown-formatted version for readability:
- Top 10 experiments
- Bottom 10 experiments
- Summary statistics

### 3. `what_worked.md`

Narrative summary of techniques that improved performance:
- Evidence (experiments, metrics)
- Effect size
- Conditions under which it worked

### 4. `what_didnt_work.md`

Narrative summary of techniques that hurt or had no effect:
- Evidence
- Why it may have failed
- Whether worth revisiting

### 5. `optimization_findings.md`

Focused on training/optimization:
- Learning rate findings
- Scheduler findings
- Batch size effects
- Gradient clipping
- Epoch requirements
- Early stopping effectiveness

### 6. `architecture_findings.md`

Focused on model architecture:
- Attention mechanisms
- Input embeddings
- Hidden size effects
- Learnable h0
- Activation functions
- Normalization

### 7. `input_design_findings.md`

Focused on input structure:
- Factor assignment
- Timing
- Sparsity
- RF-matching

### 8. `untested_combinations.md`

Promising combinations not yet tried:
- Based on individual technique success
- Unexplored parameter ranges
- Architectural variants

### 9. Visualizations (`figures/`)

| Figure | Description |
|--------|-------------|
| `fit_quality_over_time.png` | Timeline of best val_corr by phase |
| `parameter_vs_performance.png` | Scatter plots of key params vs performance |
| `configuration_heatmap.png` | Matrix of tested configurations |

### 10. `AUDIT_REPORT.md`

Executive summary:
- Total experiments catalogued
- Best performance achieved
- Top 5 key findings
- Open questions
- Prioritized recommendations

---

## Execution Steps

### Step 1: Data Collection
1. Parse all config.json files
2. Parse all results.json/summary.json files
3. Extract from history.json where needed
4. Manual review of spec files for context

### Step 2: Table Creation
1. Create DataFrame with all fields
2. Handle missing values appropriately
3. Export to CSV
4. Create markdown summary

### Step 3: Analysis
1. Compute rankings
2. Compute technique-specific statistics
3. Identify patterns and anomalies
4. Cross-validate with spec file narratives

### Step 4: Synthesis
1. Write narrative documents
2. Identify untested combinations
3. Create visualizations
4. Write executive summary

### Step 5: Validation
1. Cross-check numbers with source files
2. Ensure consistency across documents
3. Flag any ambiguities or data quality issues

---

## Quality Considerations

### Data Quality Flags

- **INCOMPLETE**: Missing config or results
- **FAILED**: Experiment crashed or produced invalid results
- **DIFFERENT_DATASET**: Uses different data file (not directly comparable)
- **EARLY_STOPPED**: Stopped before max_epochs
- **CONVERGED**: Reached patience limit (likely optimal)

### Comparability Notes

- Phase 2-5 experiments use Newton_08_15_2025_SC (54 neurons)
- Replication uses Newton_08_13, 08_14, 08_15 (varying neurons)
- Input restructuring uses different neuron counts (42 neurons)
- Early tests may use different datasets

---

## Timeline

| Step | Estimated Output |
|------|------------------|
| Data Collection | experiments_table.csv |
| Analysis | Narrative documents |
| Visualization | figures/ directory |
| Executive Summary | AUDIT_REPORT.md |

---

## Success Criteria

1. All ~120 experiments catalogued with available configuration and results
2. Clear identification of best-performing configuration (expected: 0.4070-0.4121)
3. Actionable recommendations for next experiments
4. Comprehensive coverage of what worked/didn't work
5. Identification of promising untested combinations
