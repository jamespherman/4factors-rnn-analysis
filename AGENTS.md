# AGENTS.md - Instructions for AI Coding Assistants

## Project Overview

This repository fits excitatory-inhibitory RNNs to superior colliculus (SC) neural recordings to test whether putative interneurons mediate factor-specific attentional competition.

**Core Hypothesis**: SC putative interneurons are task-modulated but non-discriminating for factors (reward, salience, probability, identity). If local lateral inhibition resolved competition, I→E weights should show factor-specific structure. Finding unstructured weights supports competition resolution via SC-external loops.

## Repository Structure

```
4factors-rnn-analysis/
├── AGENTS.md                 # This file
├── README.md                 # Project overview
├── requirements.txt          # Python dependencies
├── specs/                    # Specification documents
│   ├── DATA_SPEC.md         # Data format and export instructions
│   ├── ARCHITECTURE_SPEC.md # RNN model architecture
│   ├── TRAINING_SPEC.md     # Loss functions and optimization
│   └── ANALYSIS_SPEC.md     # Weight analysis procedures
├── data/                     # Exported .mat files (not tracked in git)
│   └── README.md            # Instructions for data location
├── src/
│   ├── data_loading.py      # Load and preprocess .mat files
│   ├── models/
│   │   ├── __init__.py
│   │   └── ei_rnn.py        # E-I RNN model definition
│   ├── training/
│   │   ├── __init__.py
│   │   ├── losses.py        # L_neuron and L_trial loss functions
│   │   └── trainer.py       # Training loop
│   └── analysis/
│       ├── __init__.py
│       └── weight_analysis.py  # I→E weight structure analysis
├── notebooks/                # Jupyter notebooks for exploration
└── results/                  # Trained models and analysis outputs
```

## Key Technical Decisions

### Framework
- **PyTorch** for RNN implementation and gradient computation
- **scipy.io** for loading .mat files from MATLAB pipeline
- Rate-based RNN (not spiking) - Sourmpis et al. (2026) showed spiking doesn't improve perturbation prediction

### Model Constraints (Critical for Interpretability)
1. **Dale's Law**: Enforced via `W = |W_raw| * sign_mask` where sign_mask is fixed based on neuron type
2. **4:1 E:I Ratio**: 80% excitatory, 20% inhibitory units
3. **Local-only Inhibition**: I units project only to other units within SC (no output projections)
4. **One-to-one Mapping**: Each recorded neuron maps to exactly one RNN unit

### Neuron Classification (from MATLAB pipeline)
- **Classic SC neurons → Excitatory units**: Significant visual/delay/saccade response, FR>5sp/s, no bilateral response, not sparse
- **Putative interneurons → Inhibitory units**: Task-modulated, FR>5sp/s, fails classic criteria

### Data Source
- Primary data location: `OneDrive/Neuronal Data Analysis/4factors-analysis-pipeline/data/processed/`
- Initial test session: `Feynman_08_15_2025_SC`
- Export script generates .mat files consumed by this Python repo

## Implementation Guidelines

### When Writing Code
1. Always read the relevant SPEC document first
2. Use type hints for all function signatures
3. Include docstrings with parameter descriptions
4. Add assertions for input validation (shapes, ranges, etc.)
5. Use descriptive variable names matching the specs (e.g., `W_rec`, `L_neuron`, `L_trial`)

### When Debugging
1. Check tensor shapes at each step - shape mismatches are the most common bug
2. Verify Dale's law constraints are maintained after each optimization step
3. Monitor loss components separately (L_neuron vs L_trial)
4. Use small subsets of data for quick iteration

### When Running Experiments
1. Save all hyperparameters with each trained model
2. Use fixed random seeds for reproducibility
3. Log training curves for both loss components
4. Validate on held-out trials before running analysis

## Critical Implementation Notes

### Loss Function (from Sourmpis et al. 2026)
```python
L = L_neuron + L_trial

# L_neuron: MSE of trial-averaged firing rates (PSTH)
# L_trial: Distance between single-trial population trajectories
```

The L_trial "trial-matching" loss is essential - it captures single-trial variability, not just averages. See TRAINING_SPEC.md for exact formulation.

### Weight Initialization
Following Sourmpis et al.: Initialize with balanced E/I input (sum of excitatory ≈ sum of inhibitory incoming weights), then normalize by spectral radius.

### Key Analysis
After fitting, examine `W_rec[I_units, E_units]` - the weights from inhibitory to excitatory units. The central question: Are these weights structured by the factor selectivity of the target E units?

## Dependencies

```
torch>=2.0
numpy
scipy
matplotlib
scikit-learn  # for decoding analyses
tqdm
```

## Contact

James Herman - Principal Investigator
Herman Lab, University of Pittsburgh
