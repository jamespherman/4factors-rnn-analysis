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
