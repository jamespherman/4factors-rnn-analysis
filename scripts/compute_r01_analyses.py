#!/usr/bin/env python3
"""
Compute R01 Figure Analyses.

This script computes all analyses needed for the R01 figure:
- Panel B: I→E connectivity vs E neuron factor selectivity
- Panel C: Model I neuron vs recorded interneuron selectivity
- Panel D: Input-potent / input-null subspace decomposition
- Panel E: Recorded neuron ROC selectivity scatter

Aggregates data across 3 Newton recording sessions.

Author: Claude Code
Date: 2025-01-25
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy import stats
from scipy.linalg import null_space, qr
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
import h5py
import json
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_session, load_mat_file


# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path("/Users/jph/Library/CloudStorage/OneDrive-UniversityofPittsburgh/Code/4factors-rnn-analysis")

# Session files
SESSIONS = {
    '08_13': BASE_DIR / 'data/rnn_export_Newton_08_13_2025_SC.mat',
    '08_14': BASE_DIR / 'data/rnn_export_Newton_08_14_2025_SC.mat',
    '08_15': BASE_DIR / 'data/rnn_export_Newton_08_15_2025_SC.mat',
}

# Model directories (trained full E/I models)
MODEL_DIRS = {
    '08_13': BASE_DIR / 'results/replication/Newton_08_13_2025_SC',
    '08_14': BASE_DIR / 'results/replication/Newton_08_14_2025_SC',
    '08_15': BASE_DIR / 'results/final_model',
}

# E-only model directories
E_ONLY_DIRS = {
    '08_13': BASE_DIR / 'results/e_only_model/session_08_13',
    '08_14': BASE_DIR / 'results/e_only_model/session_08_14',
    '08_15': BASE_DIR / 'results/e_only_model/session_08_15',
}

# Output directory
OUTPUT_DIR = BASE_DIR / 'results/r01_figure/panel_data'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Helper Functions
# ============================================================================

def load_session_data(session_id):
    """Load raw data for a session."""
    filepath = SESSIONS[session_id]
    data = load_mat_file(str(filepath))
    return data


def load_model_weights(model_dir):
    """Load trained model weights."""
    weights_dir = model_dir / 'weights'
    if not weights_dir.exists():
        return None

    W_rec = np.load(weights_dir / 'W_rec.npy')
    return {'W_rec': W_rec}


def get_neuron_counts(data):
    """Get E/I neuron counts from data."""
    neuron_type = data['neuron_type']
    n_e = np.sum(neuron_type == 1)
    n_i = np.sum(neuron_type == 2)
    return n_e, n_i


def compute_factor_selectivity_regression(firing_rates, trial_labels, factor_names):
    """
    Compute factor selectivity using multiple regression (partial eta-squared).

    Args:
        firing_rates: [n_trials, n_neurons] - Mean firing rate per trial
        trial_labels: dict with factor labels for each trial
        factor_names: list of factor names

    Returns:
        selectivity: [n_neurons, n_factors] - Partial eta-squared
        pvalues: [n_neurons, n_factors] - P-values
    """
    n_trials, n_neurons = firing_rates.shape
    n_factors = len(factor_names)

    selectivity = np.zeros((n_neurons, n_factors))
    pvalues = np.ones((n_neurons, n_factors))

    for i in range(n_neurons):
        y = firing_rates[:, i]

        # Build design matrix
        X = np.column_stack([trial_labels[f] for f in factor_names])

        # Standardize X
        X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        X_std = sm.add_constant(X_std)

        try:
            model = sm.OLS(y, X_std).fit()

            for j, factor in enumerate(factor_names):
                t_stat = model.tvalues[j + 1]
                df_resid = model.df_resid
                selectivity[i, j] = t_stat**2 / (t_stat**2 + df_resid)
                pvalues[i, j] = model.pvalues[j + 1]
        except Exception:
            selectivity[i, :] = np.nan
            pvalues[i, :] = np.nan

    return selectivity, pvalues


def compute_roc_selectivity(firing_rates, trial_labels, factor):
    """
    Compute ROC-based selectivity for each neuron.

    Args:
        firing_rates: [n_trials, n_neurons]
        trial_labels: binary array for factor (0/1)
        factor: name of factor (for high/low coding)

    Returns:
        selectivity: [n_neurons] - Selectivity index: 2*(AUC - 0.5), range [-1, 1]
    """
    n_trials, n_neurons = firing_rates.shape
    selectivity = np.zeros(n_neurons)

    # Ensure binary labels
    labels = np.array(trial_labels)
    if len(np.unique(labels)) < 2:
        return selectivity

    for i in range(n_neurons):
        rates = firing_rates[:, i]

        # Remove NaN
        valid = ~np.isnan(rates)
        if valid.sum() < 10:
            selectivity[i] = 0
            continue

        try:
            auc = roc_auc_score(labels[valid], rates[valid])
            selectivity[i] = 2 * (auc - 0.5)
        except Exception:
            selectivity[i] = 0

    return selectivity


# ============================================================================
# Panel B: I→E Connectivity vs Factor Selectivity
# ============================================================================

def compute_panel_b_data():
    """
    Compute I→E connectivity vs E neuron factor selectivity.

    For each session:
    - Load model weights
    - Extract I→E weight submatrix
    - Compute total inhibitory input to each E neuron
    - Compute factor selectivity for each E neuron
    - Correlate

    Aggregate across sessions.
    """
    print("\n" + "=" * 60)
    print("Computing Panel B: I→E Connectivity vs Factor Selectivity")
    print("=" * 60)

    all_e_selectivity_reward = []
    all_e_selectivity_salience = []
    all_i_input = []
    session_labels = []

    for session_id, data_path in SESSIONS.items():
        print(f"\nProcessing session {session_id}...")

        # Load data
        data = load_session_data(session_id)
        n_e, n_i = get_neuron_counts(data)
        print(f"  {n_e} E neurons, {n_i} I neurons")

        # Load model weights
        model_dir = MODEL_DIRS[session_id]
        weights = load_model_weights(model_dir)

        if weights is None:
            print(f"  WARNING: No weights found for session {session_id}")
            continue

        W_rec = weights['W_rec']
        n_model_e = W_rec.shape[0] - n_i  # Model may have hidden E neurons

        # Extract I→E weights: W_rec[E_targets, I_sources]
        # With hidden E neurons, we only care about recorded E neurons (first n_e)
        W_IE = W_rec[:n_e, n_model_e:n_model_e + n_i]  # [n_e, n_i]

        # Total inhibitory input to each E neuron (sum of absolute weights)
        total_i_input = np.sum(np.abs(W_IE), axis=1)  # [n_e]

        # Get firing rates for E neurons
        firing_rates = data['firing_rates']  # [n_neurons, n_time, n_trials]
        mean_rates = firing_rates.mean(axis=1).T  # [n_trials, n_neurons]
        e_rates = mean_rates[:, :n_e]  # Only E neurons

        # Get trial labels
        trial_labels = {
            'reward': data['trial_reward'],
            'salience': data.get('trial_salience', np.zeros(mean_rates.shape[0])),
        }

        # Compute factor selectivity for E neurons
        selectivity, _ = compute_factor_selectivity_regression(
            e_rates, trial_labels, ['reward', 'salience']
        )

        # Store
        all_e_selectivity_reward.extend(selectivity[:, 0])
        all_e_selectivity_salience.extend(selectivity[:, 1])
        all_i_input.extend(total_i_input)
        session_labels.extend([session_id] * n_e)

        print(f"  Added {n_e} E neurons")

    # Convert to arrays
    all_e_selectivity_reward = np.array(all_e_selectivity_reward)
    all_e_selectivity_salience = np.array(all_e_selectivity_salience)
    all_i_input = np.array(all_i_input)
    session_labels = np.array(session_labels)

    # Remove NaN
    valid_reward = ~np.isnan(all_e_selectivity_reward)
    valid_salience = ~np.isnan(all_e_selectivity_salience)

    # Compute correlations
    r_reward, p_reward = stats.pearsonr(
        all_e_selectivity_reward[valid_reward],
        all_i_input[valid_reward]
    )
    r_salience, p_salience = stats.pearsonr(
        all_e_selectivity_salience[valid_salience],
        all_i_input[valid_salience]
    )

    print(f"\n  Reward selectivity vs I input: r = {r_reward:.4f}, p = {p_reward:.4f}")
    print(f"  Salience selectivity vs I input: r = {r_salience:.4f}, p = {p_salience:.4f}")

    # Save data
    panel_b_data = {
        'e_selectivity_reward': all_e_selectivity_reward,
        'e_selectivity_salience': all_e_selectivity_salience,
        'total_i_input': all_i_input,
        'session': session_labels,
        'stats': {
            'r_reward': r_reward,
            'p_reward': p_reward,
            'r_salience': r_salience,
            'p_salience': p_salience,
            'n_e_neurons': len(all_i_input),
        }
    }

    np.savez(OUTPUT_DIR / 'panel_b_data.npz', **{
        'e_selectivity_reward': all_e_selectivity_reward,
        'e_selectivity_salience': all_e_selectivity_salience,
        'total_i_input': all_i_input,
        'session': session_labels,
    })

    with open(OUTPUT_DIR / 'panel_b_stats.json', 'w') as f:
        json.dump(panel_b_data['stats'], f, indent=2)

    print(f"  Saved to {OUTPUT_DIR / 'panel_b_data.npz'}")

    return panel_b_data


# ============================================================================
# Panel C: Model I Neurons vs Recorded Interneurons
# ============================================================================

def compute_panel_c_data():
    """
    Compare model I neurons (from E-only model) with recorded interneurons.

    For each session:
    - Load E-only model I neuron PSTHs
    - Load recorded interneuron firing rates
    - Compute factor selectivity for both
    - Compare distributions
    """
    print("\n" + "=" * 60)
    print("Computing Panel C: Model I vs Recorded Interneurons")
    print("=" * 60)

    all_model_i_selectivity = []
    all_recorded_i_selectivity = []
    session_labels_model = []
    session_labels_recorded = []

    for session_id, data_path in SESSIONS.items():
        print(f"\nProcessing session {session_id}...")

        # Load data
        data = load_session_data(session_id)
        n_e, n_i = get_neuron_counts(data)

        # Check for E-only model results
        e_only_dir = E_ONLY_DIRS[session_id]
        if not (e_only_dir / 'model_i_psth.npy').exists():
            print(f"  WARNING: No E-only model found for {session_id}")
            continue

        # Load E-only model I neuron PSTHs
        model_i_psth = np.load(e_only_dir / 'model_i_psth.npy')  # [time, n_i_model]
        n_i_model = model_i_psth.shape[1]
        print(f"  Model has {n_i_model} I neurons")

        # Get firing rates
        firing_rates = data['firing_rates']  # [n_neurons, n_time, n_trials]
        n_neurons, n_time, n_trials = firing_rates.shape

        # Mean rates per trial
        mean_rates = firing_rates.mean(axis=1).T  # [n_trials, n_neurons]

        # Recorded I neurons are after E neurons
        recorded_i_rates = mean_rates[:, n_e:n_e + n_i]  # [n_trials, n_i]

        # Get trial labels
        trial_labels = {
            'reward': data['trial_reward'],
            'salience': data.get('trial_salience', np.zeros(n_trials)),
        }

        # Compute selectivity for recorded I neurons
        recorded_sel, _ = compute_factor_selectivity_regression(
            recorded_i_rates, trial_labels, ['reward', 'salience']
        )

        # For model I neurons, we need to compute selectivity from the PSTHs
        # The model_i_psth is trial-averaged, so we need per-trial data
        if (e_only_dir / 'model_i_rates.npy').exists():
            model_i_rates = np.load(e_only_dir / 'model_i_rates.npy')  # [n_val_trials, time, n_i]

            # Match trial labels to validation trials (last 20%)
            n_val = model_i_rates.shape[0]
            val_start = n_trials - n_val

            val_trial_labels = {
                'reward': trial_labels['reward'][val_start:],
                'salience': trial_labels['salience'][val_start:],
            }

            # Mean over time
            model_i_mean = model_i_rates.mean(axis=1)  # [n_val_trials, n_i]

            model_sel, _ = compute_factor_selectivity_regression(
                model_i_mean, val_trial_labels, ['reward', 'salience']
            )
        else:
            print(f"  WARNING: No model_i_rates found, using PSTH variance as proxy")
            # Use temporal variance as a proxy for selectivity
            model_sel = np.var(model_i_psth, axis=0).reshape(-1, 1)  # [n_i, 1]
            model_sel = np.hstack([model_sel, model_sel])  # [n_i, 2]

        # Combine reward and salience selectivity (use max)
        recorded_combined = np.nanmax(np.abs(recorded_sel), axis=1)
        model_combined = np.nanmax(np.abs(model_sel), axis=1)

        all_recorded_i_selectivity.extend(recorded_combined)
        all_model_i_selectivity.extend(model_combined)
        session_labels_recorded.extend([session_id] * len(recorded_combined))
        session_labels_model.extend([session_id] * len(model_combined))

        print(f"  Added {n_i} recorded I neurons, {n_i_model} model I neurons")

    # Convert to arrays
    all_model_i_selectivity = np.array(all_model_i_selectivity)
    all_recorded_i_selectivity = np.array(all_recorded_i_selectivity)

    # Remove NaN
    valid_model = ~np.isnan(all_model_i_selectivity)
    valid_recorded = ~np.isnan(all_recorded_i_selectivity)

    # Compare distributions
    ks_stat, ks_pval = stats.ks_2samp(
        all_model_i_selectivity[valid_model],
        all_recorded_i_selectivity[valid_recorded]
    )

    # Effect size (Cohen's d)
    mean_model = np.nanmean(all_model_i_selectivity)
    mean_recorded = np.nanmean(all_recorded_i_selectivity)
    std_pooled = np.sqrt((np.nanvar(all_model_i_selectivity) + np.nanvar(all_recorded_i_selectivity)) / 2)
    cohens_d = (mean_model - mean_recorded) / (std_pooled + 1e-8)

    print(f"\n  Model I selectivity: mean = {mean_model:.4f}, n = {valid_model.sum()}")
    print(f"  Recorded I selectivity: mean = {mean_recorded:.4f}, n = {valid_recorded.sum()}")
    print(f"  KS test: D = {ks_stat:.4f}, p = {ks_pval:.4f}")
    print(f"  Cohen's d = {cohens_d:.4f}")

    # Save data
    panel_c_data = {
        'model_i_selectivity': all_model_i_selectivity,
        'recorded_i_selectivity': all_recorded_i_selectivity,
        'stats': {
            'mean_model': mean_model,
            'mean_recorded': mean_recorded,
            'ks_stat': ks_stat,
            'ks_pval': ks_pval,
            'cohens_d': cohens_d,
            'n_model': int(valid_model.sum()),
            'n_recorded': int(valid_recorded.sum()),
        }
    }

    np.savez(OUTPUT_DIR / 'panel_c_data.npz',
             model_i_selectivity=all_model_i_selectivity,
             recorded_i_selectivity=all_recorded_i_selectivity)

    with open(OUTPUT_DIR / 'panel_c_stats.json', 'w') as f:
        json.dump(panel_c_data['stats'], f, indent=2)

    print(f"  Saved to {OUTPUT_DIR / 'panel_c_data.npz'}")

    return panel_c_data


# ============================================================================
# Panel D: Input-Potent / Input-Null Subspace Decomposition
# ============================================================================

def compute_subspace_variance(neural_activity, inputs, trial_labels, factor):
    """
    Compute variance explained by factor in input-potent and input-null subspaces.

    Based on Kaufman et al. (2014) Nat Neurosci.

    Args:
        neural_activity: [n_trials, n_time, n_neurons] - Population activity
        inputs: [n_trials, n_time, n_inputs] - Input signals
        trial_labels: [n_trials] - Factor labels (0/1)
        factor: name of factor

    Returns:
        var_potent: Proportion of factor variance in input-potent subspace
        var_null: Proportion of factor variance in input-null subspace
    """
    n_trials, n_time, n_neurons = neural_activity.shape
    _, _, n_inputs = inputs.shape

    # 1. Compute trial-averaged activity and inputs
    X_mean = neural_activity.mean(axis=0)  # [n_time, n_neurons]
    U_mean = inputs.mean(axis=0)  # [n_time, n_inputs]

    # 2. Regress activity onto inputs: X = B @ U
    # B is [n_neurons, n_inputs]
    # Solve: X.T = B @ U.T => X.T is [n_neurons, n_time], U.T is [n_inputs, n_time]
    # Least squares: B = X.T @ pinv(U.T)
    U_pinv = np.linalg.pinv(U_mean.T)  # [n_time, n_inputs]
    B = X_mean.T @ U_pinv  # [n_neurons, n_inputs]

    # 3. Get input-potent subspace via QR decomposition of B
    # Q_potent spans the column space of B
    if np.linalg.matrix_rank(B) == 0:
        # No input-potent subspace
        return 0.5, 0.5

    Q_potent, R = qr(B, mode='economic')  # Q_potent is [n_neurons, k]
    k = Q_potent.shape[1]

    # 4. Get input-null subspace (orthogonal complement)
    # Use null space of B.T
    Q_null = null_space(B.T)  # [n_neurons, n_neurons - k]

    if Q_null.shape[1] == 0:
        # All activity is input-potent
        return 1.0, 0.0

    # 5. Compute factor-related variance in each subspace
    # For each trial, project onto subspaces and compute variance by factor
    labels = np.array(trial_labels)
    high_idx = labels == 1
    low_idx = labels == 0

    if high_idx.sum() < 2 or low_idx.sum() < 2:
        return 0.5, 0.5

    # Flatten activity: [n_trials, n_time * n_neurons]
    X_flat = neural_activity.reshape(n_trials, -1)

    # Actually, we want to project activity at each timepoint
    # Let's compute mean activity per condition
    X_high = neural_activity[high_idx].mean(axis=0)  # [n_time, n_neurons]
    X_low = neural_activity[low_idx].mean(axis=0)  # [n_time, n_neurons]

    # Factor-related component
    X_factor = X_high - X_low  # [n_time, n_neurons]

    # Project onto subspaces
    X_factor_potent = X_factor @ Q_potent  # [n_time, k]
    X_factor_null = X_factor @ Q_null  # [n_time, n_neurons - k]

    # Variance in each subspace
    var_potent = np.var(X_factor_potent)
    var_null = np.var(X_factor_null)

    total_var = var_potent + var_null
    if total_var < 1e-10:
        return 0.5, 0.5

    return var_potent / total_var, var_null / total_var


def compute_panel_d_data():
    """
    Compute input-potent / input-null subspace decomposition.

    For each session:
    - Load neural activity and inputs
    - Compute subspace decomposition
    - Compute factor variance in each subspace
    """
    print("\n" + "=" * 60)
    print("Computing Panel D: Subspace Variance Decomposition")
    print("=" * 60)

    results = {
        'reward': {'potent': [], 'null': []},
        'salience': {'potent': [], 'null': []},
    }
    session_ids = []

    for session_id, data_path in SESSIONS.items():
        print(f"\nProcessing session {session_id}...")

        # Load data
        data = load_session_data(session_id)
        dataset = load_session(str(data_path), validate=False)

        n_e, n_i = get_neuron_counts(data)
        n_trials = int(data['n_trials'])

        # Get neural activity [n_trials, n_time, n_neurons]
        firing_rates = data['firing_rates']  # [n_neurons, n_time, n_trials]
        neural_activity = firing_rates.transpose(2, 1, 0)  # [n_trials, n_time, n_neurons]

        # Use only E neurons (recorded)
        neural_activity_e = neural_activity[:, :, :n_e]

        # Get inputs from dataset
        all_trials = dataset.get_all_trials()
        inputs = all_trials['inputs'].numpy()  # [n_trials, n_time, n_inputs]

        # Get trial labels
        trial_labels_reward = data['trial_reward']
        trial_labels_salience_raw = data.get('trial_salience', np.zeros(n_trials))

        # Binarize salience: High (2) -> 1, Low (0) -> 0, exclude Medium (1)
        salience_binary_mask = (trial_labels_salience_raw == 0) | (trial_labels_salience_raw == 2)
        trial_labels_salience = (trial_labels_salience_raw == 2).astype(float)

        # Compute for reward factor
        var_potent_reward, var_null_reward = compute_subspace_variance(
            neural_activity_e, inputs, trial_labels_reward, 'reward'
        )

        # Compute for salience factor (using only high/low trials)
        if salience_binary_mask.sum() > 20:
            var_potent_salience, var_null_salience = compute_subspace_variance(
                neural_activity_e[salience_binary_mask],
                inputs[salience_binary_mask],
                trial_labels_salience[salience_binary_mask],
                'salience'
            )
        else:
            var_potent_salience, var_null_salience = 0.5, 0.5

        results['reward']['potent'].append(var_potent_reward)
        results['reward']['null'].append(var_null_reward)
        results['salience']['potent'].append(var_potent_salience)
        results['salience']['null'].append(var_null_salience)
        session_ids.append(session_id)

        print(f"  Reward: potent = {var_potent_reward:.3f}, null = {var_null_reward:.3f}")
        print(f"  Salience: potent = {var_potent_salience:.3f}, null = {var_null_salience:.3f}")

    # Compute statistics
    panel_d_data = {
        'reward_potent_mean': np.mean(results['reward']['potent']),
        'reward_potent_sem': stats.sem(results['reward']['potent']),
        'reward_null_mean': np.mean(results['reward']['null']),
        'reward_null_sem': stats.sem(results['reward']['null']),
        'salience_potent_mean': np.mean(results['salience']['potent']),
        'salience_potent_sem': stats.sem(results['salience']['potent']),
        'salience_null_mean': np.mean(results['salience']['null']),
        'salience_null_sem': stats.sem(results['salience']['null']),
        'n_sessions': len(session_ids),
    }

    # Save raw data
    np.savez(OUTPUT_DIR / 'panel_d_data.npz',
             reward_potent=results['reward']['potent'],
             reward_null=results['reward']['null'],
             salience_potent=results['salience']['potent'],
             salience_null=results['salience']['null'],
             sessions=session_ids)

    with open(OUTPUT_DIR / 'panel_d_stats.json', 'w') as f:
        json.dump(panel_d_data, f, indent=2)

    print(f"\n  Summary:")
    print(f"  Reward: potent = {panel_d_data['reward_potent_mean']:.3f} +/- {panel_d_data['reward_potent_sem']:.3f}")
    print(f"  Salience: potent = {panel_d_data['salience_potent_mean']:.3f} +/- {panel_d_data['salience_potent_sem']:.3f}")
    print(f"  Saved to {OUTPUT_DIR / 'panel_d_data.npz'}")

    return panel_d_data


# ============================================================================
# Panel E: Recorded Neuron ROC Scatter
# ============================================================================

def compute_panel_e_data():
    """
    Compute ROC selectivity for all recorded neurons.

    For each session:
    - Compute reward ROC selectivity
    - Compute salience ROC selectivity
    - Label E vs I neurons
    """
    print("\n" + "=" * 60)
    print("Computing Panel E: Recorded Neuron ROC Scatter")
    print("=" * 60)

    all_reward_selectivity = []
    all_salience_selectivity = []
    all_neuron_types = []
    session_labels = []

    for session_id, data_path in SESSIONS.items():
        print(f"\nProcessing session {session_id}...")

        # Load data
        data = load_session_data(session_id)
        n_e, n_i = get_neuron_counts(data)
        n_neurons = n_e + n_i
        n_trials = int(data['n_trials'])

        # Get firing rates
        firing_rates = data['firing_rates']  # [n_neurons, n_time, n_trials]
        mean_rates = firing_rates.mean(axis=1).T  # [n_trials, n_neurons]

        # Get trial labels
        trial_reward = data['trial_reward']
        trial_salience_raw = data.get('trial_salience', np.zeros(n_trials))

        # Binarize salience: High (2) vs Low (0), excluding Medium (1)
        # This gives a cleaner high-vs-low comparison
        salience_binary_mask = (trial_salience_raw == 0) | (trial_salience_raw == 2)
        trial_salience_binary = (trial_salience_raw == 2).astype(float)  # 1 for high, 0 for low

        # Compute ROC selectivity
        reward_sel = compute_roc_selectivity(mean_rates, trial_reward, 'reward')

        # For salience, use only high/low trials
        if salience_binary_mask.sum() > 20:
            salience_sel = compute_roc_selectivity(
                mean_rates[salience_binary_mask],
                trial_salience_binary[salience_binary_mask],
                'salience'
            )
        else:
            salience_sel = np.zeros(mean_rates.shape[1])

        # Neuron types
        neuron_types = ['E'] * n_e + ['I'] * n_i

        all_reward_selectivity.extend(reward_sel)
        all_salience_selectivity.extend(salience_sel)
        all_neuron_types.extend(neuron_types)
        session_labels.extend([session_id] * n_neurons)

        print(f"  Added {n_e} E and {n_i} I neurons")

    # Convert to arrays
    all_reward_selectivity = np.array(all_reward_selectivity)
    all_salience_selectivity = np.array(all_salience_selectivity)
    all_neuron_types = np.array(all_neuron_types)
    session_labels = np.array(session_labels)

    # Separate E and I
    e_mask = all_neuron_types == 'E'
    i_mask = all_neuron_types == 'I'

    # Statistics
    print(f"\n  E neurons: n = {e_mask.sum()}")
    print(f"    Reward selectivity: mean = {all_reward_selectivity[e_mask].mean():.4f}")
    print(f"    Salience selectivity: mean = {all_salience_selectivity[e_mask].mean():.4f}")

    print(f"  I neurons: n = {i_mask.sum()}")
    print(f"    Reward selectivity: mean = {all_reward_selectivity[i_mask].mean():.4f}")
    print(f"    Salience selectivity: mean = {all_salience_selectivity[i_mask].mean():.4f}")

    # Save data
    np.savez(OUTPUT_DIR / 'panel_e_data.npz',
             reward_selectivity=all_reward_selectivity,
             salience_selectivity=all_salience_selectivity,
             neuron_type=all_neuron_types,
             session=session_labels)

    panel_e_stats = {
        'n_e': int(e_mask.sum()),
        'n_i': int(i_mask.sum()),
        'e_reward_mean': float(all_reward_selectivity[e_mask].mean()),
        'e_salience_mean': float(all_salience_selectivity[e_mask].mean()),
        'i_reward_mean': float(all_reward_selectivity[i_mask].mean()),
        'i_salience_mean': float(all_salience_selectivity[i_mask].mean()),
    }

    with open(OUTPUT_DIR / 'panel_e_stats.json', 'w') as f:
        json.dump(panel_e_stats, f, indent=2)

    print(f"  Saved to {OUTPUT_DIR / 'panel_e_data.npz'}")

    return {
        'reward_selectivity': all_reward_selectivity,
        'salience_selectivity': all_salience_selectivity,
        'neuron_type': all_neuron_types,
        'session': session_labels,
        'stats': panel_e_stats,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all analyses."""
    print("=" * 70)
    print("R01 FIGURE ANALYSES")
    print("=" * 70)

    # Panel B
    panel_b_data = compute_panel_b_data()

    # Panel C (requires E-only models - may skip if not trained yet)
    try:
        panel_c_data = compute_panel_c_data()
    except Exception as e:
        print(f"\nWARNING: Panel C computation failed: {e}")
        print("Train E-only models first using train_e_only_model.py")
        panel_c_data = None

    # Panel D
    panel_d_data = compute_panel_d_data()

    # Panel E
    panel_e_data = compute_panel_e_data()

    print("\n" + "=" * 70)
    print("ANALYSES COMPLETE")
    print("=" * 70)
    print(f"\nOutput saved to: {OUTPUT_DIR}")

    return {
        'panel_b': panel_b_data,
        'panel_c': panel_c_data,
        'panel_d': panel_d_data,
        'panel_e': panel_e_data,
    }


if __name__ == "__main__":
    results = main()
