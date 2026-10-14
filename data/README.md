# Data Directory

This directory contains exported `.mat` files from the MATLAB pipeline.

## How to populate

1. Run the MATLAB export script from `4factors-analysis-pipeline`:

```matlab
cd /path/to/4factors-analysis-pipeline
addpath(genpath('scripts'))
export_for_rnn('Feynman_08_15_2025_SC', '/path/to/4factors-rnn-analysis/data/')
```

2. Verify the export:

```python
from src.data_loader import load_session
dataset = load_session('data/rnn_export_Feynman_08_15_2025_SC.mat')
```

## Expected files

After export, you should have:
- `rnn_export_Feynman_08_15_2025_SC.mat`

## Data format

See `specs/DATA_SPEC.md` for detailed format documentation.

## Export versions (2026-10-14)

- `rnn_export_*.mat` here: corrected exports, pipeline 1.2.0, all 7 SC sessions (Feynman 08_12-15,
  Newton 08_13-15). Successful trials only (`trial_outcome` = 21), true reward times, and
  `neuron_stable_trials` [neurons x trials]; rates outside a neuron's stable period are masked
  (src/sessions.py). `trial_audit/`: the matching audit CSVs.
- `export_2026-10-14_v1.0/`: the earlier Newton exports (fixation breaks included, filtered with
  their audit CSVs). Every result computed before the corrected exports used these; rebuild them
  with `FOURFACTORS_EXPORT_DIR=data/export_2026-10-14_v1.0`.
- `export_2026-01-22_0631/`: older still.
