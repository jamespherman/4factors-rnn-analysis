"""
Data loading utilities for exported .mat files from MATLAB pipeline.

See specs/DATA_SPEC.md for format details.
"""

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Tuple, List
from pathlib import Path


def load_mat_file(filepath: str) -> dict:
    """
    Load exported .mat file from MATLAB pipeline.

    Handles both v7 (.mat) and v7.3 (HDF5) formats.

    Args:
        filepath: Path to .mat file

    Returns:
        Dictionary with all data fields as numpy arrays
    """
    try:
        # Try standard scipy.io first (for v7 and earlier)
        data = sio.loadmat(filepath, squeeze_me=True, struct_as_record=False)
        # Remove MATLAB metadata fields
        data = {k: v for k, v in data.items() if not k.startswith('__')}
    except NotImplementedError:
        # v7.3 format requires h5py
        import h5py
        data = {}
        with h5py.File(filepath, 'r') as f:
            for key in f.keys():
                if key.startswith('#'):
                    continue
                item = f[key]
                if isinstance(item, h5py.Dataset):
                    arr = item[()]
                    # HDF5 stores in column-major order, transpose if needed
                    if arr.ndim > 1:
                        arr = arr.T
                    # Convert bytes to string for string arrays
                    if arr.dtype.kind == 'O' or arr.dtype == 'uint16':
                        try:
                            arr = np.array([
                                ''.join(chr(c) for c in f[ref][:].flatten())
                                for ref in arr.flatten()
                            ]) if arr.dtype == 'O' else arr
                        except:
                            pass
                    data[key] = np.squeeze(arr)

    return data


def preprocess_firing_rates(firing_rates: np.ndarray, clip_percentile: float = 99.5) -> np.ndarray:
    """
    Preprocess firing rates: clip outliers, ensure non-negative.

    Args:
        firing_rates: [neurons, time, trials] or [trials, time, neurons] array
        clip_percentile: Percentile for outlier clipping

    Returns:
        Preprocessed firing rates
    """
    # Handle NaN values for percentile calculation
    valid_rates = firing_rates[~np.isnan(firing_rates)]

    if len(valid_rates) == 0:
        return firing_rates

    # Compute clip threshold
    clip_threshold = np.percentile(valid_rates, clip_percentile)

    # Report clipping
    n_clipped = (firing_rates > clip_threshold).sum()
    if n_clipped > 0:
        print(f"Clipping {n_clipped} values ({100*n_clipped/firing_rates.size:.3f}%) above {clip_threshold:.1f} sp/s")

    # Clip (preserve NaN)
    firing_rates = np.where(
        np.isnan(firing_rates),
        firing_rates,
        np.clip(firing_rates, 0, clip_threshold)
    )

    return firing_rates


def validate_data(data: dict) -> bool:
    """
    Validate exported data structure.
    
    Checks:
    - Required fields exist
    - Shapes are consistent
    - No unexpected NaN values
    - Neuron types are valid
    
    Returns:
        True if validation passes
    
    Raises:
        AssertionError with description if validation fails
    """
    # Required fields
    required = [
        # Neural data
        'firing_rates', 'neuron_type', 'n_trials', 'n_neurons', 'n_time_bins',
        'bin_size_ms',
        # Task inputs (all required for consistent 14-dim input)
        'input_fixation_on', 'input_target_loc', 'input_go_signal', 'input_reward_on',
        'input_eye_x', 'input_eye_y',  # Eye position (degrees, z-score normalized in Python)
        'input_is_face', 'input_is_nonface', 'input_is_bullseye',
        'input_high_salience', 'input_low_salience',
        # Trial labels
        'trial_reward', 'trial_location'
    ]

    # Optional fields
    optional = ['trial_duration_ms', 'trial_probability', 'trial_identity', 'trial_salience']
    
    for field in required:
        assert field in data, f"Missing required field: {field}"
    
    # Check dimensions
    n_neurons = int(data['n_neurons'])
    n_trials = int(data['n_trials'])
    n_bins = int(data['n_time_bins'])
    
    fr_shape = data['firing_rates'].shape
    assert fr_shape == (n_neurons, n_bins, n_trials), \
        f"firing_rates shape mismatch: {fr_shape} vs expected ({n_neurons}, {n_bins}, {n_trials})"
    
    # Check neuron types
    neuron_types = data['neuron_type']
    assert len(neuron_types) == n_neurons, "neuron_type length mismatch"
    assert set(np.unique(neuron_types)).issubset({1, 2}), \
        f"Invalid neuron types: {np.unique(neuron_types)}"
    
    # Check for NaN in firing rates (should only be in padded regions)
    # Note: Some NaN may be expected for variable-length trials
    
    # Check firing rate range (sanity check)
    fr = data['firing_rates']
    fr_valid = fr[~np.isnan(fr)]
    assert fr_valid.min() >= 0, "Negative firing rates found"
    assert fr_valid.mean() < 100, f"Suspiciously high mean FR: {fr_valid.mean()}"
    
    # Check neuron counts
    n_exc = np.sum(neuron_types == 1)
    n_inh = np.sum(neuron_types == 2)
    print(f"Validation passed:")
    print(f"  Neurons: {n_neurons} ({n_exc} E, {n_inh} I)")
    print(f"  Trials: {n_trials}")
    print(f"  Time bins: {n_bins}")
    print(f"  Mean FR: {fr_valid.mean():.2f} sp/s")
    
    return True


def construct_input_tensor(data: dict) -> np.ndarray:
    """
    Construct input tensor from data fields.

    Input channel order (14 total, matches ARCHITECTURE_SPEC.md):
        0: input_fixation_on
        1-4: input_target_loc (one-hot, 4 locations)
        5: input_go_signal
        6: input_reward_on
        7: input_eye_x (normalized, zeros if unavailable)
        8: input_eye_y (normalized, zeros if unavailable)
        9: input_is_face
        10: input_is_nonface
        11: input_is_bullseye
        12: input_high_salience
        13: input_low_salience

    Returns:
        inputs: [n_trials, n_bins, 14] array
    """
    n_trials = int(data['n_trials'])
    n_bins = int(data['n_time_bins'])

    # Collect all input channels in ARCHITECTURE_SPEC.md order
    input_channels = []

    # 0: Fixation on (transpose from [bins, trials] to [trials, bins])
    input_channels.append(data['input_fixation_on'].T)

    # 1-4: Target location (one-hot): [4, bins, trials] -> [trials, bins, 4]
    target_loc = np.transpose(data['input_target_loc'], (2, 1, 0))
    for i in range(4):
        input_channels.append(target_loc[:, :, i])

    # 5: Go signal
    input_channels.append(data['input_go_signal'].T)

    # 6: Reward on
    input_channels.append(data['input_reward_on'].T)

    # 7-8: Eye position (required, in degrees - z-score normalize here)
    # Falls back to zeros if data is all zeros/NaN (MATLAB hasn't populated yet)
    eye_x = data['input_eye_x'].T
    if np.all(np.isnan(eye_x)) or np.nanstd(eye_x) < 1e-8:
        eye_x_norm = np.zeros((n_trials, n_bins))
    else:
        eye_x_norm = (eye_x - np.nanmean(eye_x)) / (np.nanstd(eye_x) + 1e-8)

    eye_y = data['input_eye_y'].T
    if np.all(np.isnan(eye_y)) or np.nanstd(eye_y) < 1e-8:
        eye_y_norm = np.zeros((n_trials, n_bins))
    else:
        eye_y_norm = (eye_y - np.nanmean(eye_y)) / (np.nanstd(eye_y) + 1e-8)

    input_channels.append(eye_x_norm)
    input_channels.append(eye_y_norm)

    # 9-13: Target features (all required)
    input_channels.append(data['input_is_face'].T)
    input_channels.append(data['input_is_nonface'].T)
    input_channels.append(data['input_is_bullseye'].T)
    input_channels.append(data['input_high_salience'].T)
    input_channels.append(data['input_low_salience'].T)

    # Stack: [trials, bins, n_inputs]
    inputs = np.stack(input_channels, axis=2)

    # Replace NaN with 0 (for padded regions)
    inputs = np.nan_to_num(inputs, nan=0.0)

    assert inputs.shape == (n_trials, n_bins, 14), \
        f"Input shape mismatch: {inputs.shape} vs expected ({n_trials}, {n_bins}, 14)"

    return inputs.astype(np.float32)


def construct_target_tensor(data: dict, clip_outliers: bool = True) -> np.ndarray:
    """
    Construct target firing rate tensor.

    Args:
        data: Data dictionary with 'firing_rates' field
        clip_outliers: Whether to clip outlier firing rates

    Returns:
        targets: [n_trials, n_bins, n_neurons] array
    """
    # firing_rates is [n_neurons, n_bins, n_trials]
    firing_rates = data['firing_rates'].copy()

    # Clip outliers before transposing (Priority 3)
    if clip_outliers:
        firing_rates = preprocess_firing_rates(firing_rates, clip_percentile=99.5)

    # Transpose to [n_trials, n_bins, n_neurons]
    targets = np.transpose(firing_rates, (2, 1, 0))

    # Replace NaN with 0 (for padded regions)
    targets = np.nan_to_num(targets, nan=0.0)

    return targets.astype(np.float32)


def construct_mask(data: dict) -> np.ndarray:
    """
    Construct validity mask for variable-length trials.
    
    Returns:
        mask: [n_trials, n_bins] binary array (1 = valid, 0 = padding)
    """
    n_trials = int(data['n_trials'])
    n_bins = int(data['n_time_bins'])
    bin_size_ms = float(data['bin_size_ms'])
    
    mask = np.ones((n_trials, n_bins), dtype=np.float32)
    
    if 'trial_duration_ms' in data:
        trial_durations = data['trial_duration_ms']
        for i in range(n_trials):
            valid_bins = int(trial_durations[i] / bin_size_ms)
            mask[i, valid_bins:] = 0
    
    return mask


class RNNDataset(Dataset):
    """
    PyTorch Dataset for RNN training.
    
    Each item is a single trial with:
    - inputs: [n_bins, n_inputs]
    - targets: [n_bins, n_neurons]  
    - mask: [n_bins]
    - trial_info: dict with trial-level labels
    """
    
    def __init__(self, data: dict):
        """
        Initialize dataset from loaded data dictionary.
        
        Args:
            data: Dictionary from load_mat_file()
        """
        self.n_trials = int(data['n_trials'])
        self.n_neurons = int(data['n_neurons'])
        self.n_bins = int(data['n_time_bins'])
        self.bin_size_ms = float(data['bin_size_ms'])
        
        # Neuron classification
        self.neuron_type = data['neuron_type']
        self.exc_indices = np.where(self.neuron_type == 1)[0]
        self.inh_indices = np.where(self.neuron_type == 2)[0]
        self.n_exc = len(self.exc_indices)
        self.n_inh = len(self.inh_indices)
        
        # Construct tensors
        self.inputs = construct_input_tensor(data)    # [trials, bins, inputs]
        self.targets = construct_target_tensor(data)  # [trials, bins, neurons]
        self.mask = construct_mask(data)              # [trials, bins]
        
        # Trial-level labels
        self.trial_reward = data['trial_reward']
        self.trial_location = data['trial_location']
        self.trial_probability = data.get('trial_probability', np.zeros(self.n_trials))
        self.trial_identity = data.get('trial_identity', np.zeros(self.n_trials))
        self.trial_salience = data.get('trial_salience', np.zeros(self.n_trials))
        
        # Store raw data for analysis
        self._raw_data = data
    
    def __len__(self) -> int:
        return self.n_trials
    
    def __getitem__(self, idx: int) -> dict:
        return {
            'inputs': torch.tensor(self.inputs[idx], dtype=torch.float32),
            'targets': torch.tensor(self.targets[idx], dtype=torch.float32),
            'mask': torch.tensor(self.mask[idx], dtype=torch.float32),
            'trial_reward': int(self.trial_reward[idx]),
            'trial_location': int(self.trial_location[idx]),
            'trial_idx': idx
        }
    
    def get_all_trials(self) -> dict:
        """Get all trials as a single batch (for trial-matching loss)."""
        return {
            'inputs': torch.tensor(self.inputs, dtype=torch.float32),
            'targets': torch.tensor(self.targets, dtype=torch.float32),
            'mask': torch.tensor(self.mask, dtype=torch.float32),
        }
    
    def get_neuron_info(self) -> dict:
        """Get neuron classification info for model construction."""
        return {
            'n_exc': self.n_exc,
            'n_inh': self.n_inh,
            'n_total': self.n_neurons,
            'exc_indices': self.exc_indices,
            'inh_indices': self.inh_indices,
            'neuron_type': self.neuron_type
        }
    
    def get_input_dim(self) -> int:
        """Get input dimension for model construction."""
        return self.inputs.shape[2]


def train_val_split(
    dataset: RNNDataset,
    val_fraction: float = 0.2,
    stratify_by: str = 'reward',
    seed: int = 42
) -> Tuple[List[int], List[int]]:
    """
    Split dataset into train/validation indices.
    
    Args:
        dataset: RNNDataset
        val_fraction: Fraction of trials for validation
        stratify_by: Field to stratify by ('reward', 'location', etc.)
        seed: Random seed
    
    Returns:
        train_indices, val_indices
    """
    np.random.seed(seed)
    
    n_trials = len(dataset)
    indices = np.arange(n_trials)
    
    # Get stratification labels
    if stratify_by == 'reward':
        labels = dataset.trial_reward
    elif stratify_by == 'location':
        labels = dataset.trial_location
    else:
        labels = np.zeros(n_trials)
    
    # Stratified split
    train_indices = []
    val_indices = []
    
    for label in np.unique(labels):
        label_indices = indices[labels == label]
        np.random.shuffle(label_indices)
        
        n_val = int(len(label_indices) * val_fraction)
        val_indices.extend(label_indices[:n_val])
        train_indices.extend(label_indices[n_val:])
    
    return train_indices, val_indices


def load_session(
    filepath: str,
    validate: bool = True
) -> RNNDataset:
    """
    Load a session and create dataset.
    
    Args:
        filepath: Path to exported .mat file
        validate: Whether to run validation checks
    
    Returns:
        RNNDataset
    """
    print(f"Loading {filepath}...")
    data = load_mat_file(filepath)
    
    if validate:
        validate_data(data)
    
    dataset = RNNDataset(data)
    print(f"Created dataset with {len(dataset)} trials")
    
    return dataset


if __name__ == "__main__":
    # Test with synthetic data (no real file needed)
    print("Testing data loader with synthetic data...")
    
    # Create synthetic data matching expected format
    n_neurons = 50
    n_bins = 120
    n_trials = 200
    
    synthetic_data = {
        'firing_rates': np.random.rand(n_neurons, n_bins, n_trials) * 20,
        'neuron_type': np.array([1]*40 + [2]*10),  # 40 E, 10 I
        'n_trials': n_trials,
        'n_neurons': n_neurons,
        'n_time_bins': n_bins,
        'bin_size_ms': 25.0,
        'input_fixation_on': np.random.randint(0, 2, (n_bins, n_trials)),
        'input_target_loc': np.zeros((4, n_bins, n_trials)),
        'input_go_signal': np.random.randint(0, 2, (n_bins, n_trials)),
        'input_reward_on': np.random.randint(0, 2, (n_bins, n_trials)),
        'input_eye_x': np.random.randn(n_bins, n_trials),
        'input_eye_y': np.random.randn(n_bins, n_trials),
        'input_is_face': np.random.randint(0, 2, (n_bins, n_trials)),
        'input_is_nonface': np.random.randint(0, 2, (n_bins, n_trials)),
        'input_is_bullseye': np.random.randint(0, 2, (n_bins, n_trials)),
        'input_high_salience': np.random.randint(0, 2, (n_bins, n_trials)),
        'input_low_salience': np.random.randint(0, 2, (n_bins, n_trials)),
        'trial_reward': np.random.randint(0, 2, n_trials),
        'trial_location': np.random.randint(1, 5, n_trials),
        'trial_probability': np.random.randint(0, 2, n_trials),
        'trial_identity': np.random.randint(1, 4, n_trials),
        'trial_salience': np.random.randint(0, 3, n_trials),
    }
    
    # Set one-hot target location
    for i in range(n_trials):
        loc = np.random.randint(0, 4)
        synthetic_data['input_target_loc'][loc, 40:80, i] = 1
    
    # Validate
    validate_data(synthetic_data)
    
    # Create dataset
    dataset = RNNDataset(synthetic_data)
    
    print(f"\nDataset info:")
    print(f"  Trials: {len(dataset)}")
    print(f"  Input dim: {dataset.get_input_dim()}")
    
    neuron_info = dataset.get_neuron_info()
    print(f"  Neurons: {neuron_info['n_total']} ({neuron_info['n_exc']} E, {neuron_info['n_inh']} I)")
    
    # Test single item
    item = dataset[0]
    print(f"\nSingle trial shapes:")
    print(f"  inputs: {item['inputs'].shape}")
    print(f"  targets: {item['targets'].shape}")
    print(f"  mask: {item['mask'].shape}")
    
    # Test batch
    batch = dataset.get_all_trials()
    print(f"\nFull batch shapes:")
    print(f"  inputs: {batch['inputs'].shape}")
    print(f"  targets: {batch['targets'].shape}")
    
    # Test train/val split
    train_idx, val_idx = train_val_split(dataset)
    print(f"\nTrain/val split: {len(train_idx)}/{len(val_idx)}")

    # Test with all-zeros eye position (MATLAB hasn't populated yet)
    print("\nTesting with all-zeros eye position...")
    synthetic_data_zeros_eye = synthetic_data.copy()
    synthetic_data_zeros_eye['input_eye_x'] = np.zeros((n_bins, n_trials))
    synthetic_data_zeros_eye['input_eye_y'] = np.zeros((n_bins, n_trials))
    dataset_zeros_eye = RNNDataset(synthetic_data_zeros_eye)
    batch_zeros_eye = dataset_zeros_eye.get_all_trials()
    assert batch_zeros_eye['inputs'].shape == (n_trials, n_bins, 14), "Input dim should still be 14"
    # Check that eye channels (7, 8) are zeros when input is zeros
    assert torch.all(batch_zeros_eye['inputs'][:, :, 7] == 0), "Eye X should be zeros"
    assert torch.all(batch_zeros_eye['inputs'][:, :, 8] == 0), "Eye Y should be zeros"
    print("  All-zeros eye position handled correctly (remains zeros)")

    print("\nAll tests passed!")
