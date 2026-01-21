# 4factors-rnn-analysis

Excitatory-Inhibitory RNN analysis of superior colliculus neural recordings to test whether putative interneurons mediate factor-specific attentional competition.

## Scientific Question

Does the SC resolve competition between goal-directed and stimulus-driven attention **locally** via interneurons, or through **distributed** circuit interactions?

**Hypothesis**: SC putative interneurons are task-modulated but non-discriminating for factors (reward, salience, probability, identity). This suggests competition resolution occurs via SC-external loops rather than local lateral inhibition.

**Approach**: Fit E-I constrained RNNs to SC recordings, then analyze whether I→E weights show factor-specific structure.

## Quick Start

```bash
# 1. Clone and setup
git clone <repo>
cd 4factors-rnn-analysis
pip install -r requirements.txt

# 2. Configure data path
cp config.yaml.example config.yaml
# Edit config.yaml to point to your data

# 3. Export data from MATLAB (in 4factors-analysis-pipeline)
matlab -r "export_for_rnn('Feynman_08_15_2025_SC', 'path/to/output')"

# 4. Train model
python scripts/train_model.py --session Feynman_08_15_2025_SC

# 5. Analyze weights
python scripts/analyze_weights.py --model results/model_best.pt
```

## Documentation

See `specs/` for detailed specifications:

- **[DATA_SPEC.md](specs/DATA_SPEC.md)** - Data format, MATLAB export
- **[ARCHITECTURE_SPEC.md](specs/ARCHITECTURE_SPEC.md)** - RNN model architecture
- **[TRAINING_SPEC.md](specs/TRAINING_SPEC.md)** - Loss functions, optimization
- **[ANALYSIS_SPEC.md](specs/ANALYSIS_SPEC.md)** - Weight structure analysis

## Key References

- Song et al. (2016) - E-I RNN framework
- Sourmpis et al. (2026) - Trial-matching loss, biological constraints validation
- RajaKumar et al. (2021) - E-I RNN fitting methodology

## Project Structure

```
4factors-rnn-analysis/
├── AGENTS.md           # Instructions for AI coding assistants
├── README.md           # This file
├── requirements.txt    # Python dependencies
├── config.yaml         # Data paths (gitignored)
├── specs/              # Specification documents
├── src/                # Source code
├── scripts/            # Training and analysis scripts  
├── notebooks/          # Jupyter notebooks
└── results/            # Outputs (gitignored)
```

## Related Repositories

- **4factors-analysis-pipeline** (MATLAB) - Spike sorting, preprocessing, neuron classification
- This repo consumes exported `.mat` files from the MATLAB pipeline

## Author

James Herman  
Herman Lab, University of Pittsburgh  
Department of Ophthalmology / CNUP
