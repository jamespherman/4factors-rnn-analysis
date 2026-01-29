"""
Target 2 Outlier Analysis

Investigates why 32.7% of trials match to one target, identifying potential
data anomalies or dominant modes in the trial-matching loss.
"""

import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import create_model_from_data
from src.losses import smooth_temporal
from src.data_loader import load_session, train_val_split


def analyze_matching_concentration(model, data, dataset, device, output_dir):
    """Identify which target trials attract disproportionate matching."""
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        inputs = data['inputs'].to(device)
        targets = data['targets'].to(device)
        mask = data['mask'].to(device)
        
        model_rates, _ = model(inputs)
        n_recorded = targets.shape[2]
        model_rates = model_rates[:, :, :n_recorded]
        
        # Population-average activity per trial
        model_pop = model_rates.mean(dim=2)
        target_pop = targets.mean(dim=2)
        
        model_pop = model_pop * mask
        target_pop = target_pop * mask
        
        # Temporal smoothing
        kernel_size = max(1, int(32.0 / 25.0))
        model_pop = smooth_temporal(model_pop, kernel_size, dim=1)
        target_pop = smooth_temporal(target_pop, kernel_size, dim=1)
        
        # Z-score normalize
        model_mean = model_pop.mean(dim=0, keepdim=True)
        model_std = model_pop.std(dim=0, keepdim=True) + 1e-8
        model_pop_norm = (model_pop - model_mean) / model_std
        
        target_mean = target_pop.mean(dim=0, keepdim=True)
        target_std = target_pop.std(dim=0, keepdim=True) + 1e-8
        target_pop_norm = (target_pop - target_mean) / target_std
        
        # Compute pairwise distances
        distances = torch.cdist(model_pop_norm, target_pop_norm, p=2)
        
        # Soft assignment weights
        temperature = 0.1
        soft_weights = F.softmax(-distances / temperature, dim=1)
        
        # Find concentration: which target trials receive most weight?
        target_weight_sum = soft_weights.sum(dim=0).cpu().numpy()
        
        # Identify outliers (>2 std above mean)
        mean_weight = target_weight_sum.mean()
        std_weight = target_weight_sum.std()
        outlier_mask = target_weight_sum > mean_weight + 2 * std_weight
        outlier_indices = np.where(outlier_mask)[0]
        
        print(f"\n{'='*60}")
        print("MATCHING CONCENTRATION ANALYSIS")
        print(f"{'='*60}")
        print(f"Number of trials: {len(target_weight_sum)}")
        print(f"Mean weight per target trial: {mean_weight:.3f}")
        print(f"Std of weights: {std_weight:.3f}")
        print(f"Number of outlier trials (>2 std): {len(outlier_indices)}")
        
        # Analyze outlier characteristics
        results = {
            'n_trials': len(target_weight_sum),
            'mean_weight': float(mean_weight),
            'std_weight': float(std_weight),
            'outlier_indices': outlier_indices.tolist(),
            'outlier_weights': target_weight_sum[outlier_mask].tolist(),
            'outlier_details': []
        }
        
        print(f"\nOutlier Trial Details:")
        print("-" * 60)
        
        for idx in outlier_indices:
            weight = target_weight_sum[idx]
            weight_pct = weight / len(target_weight_sum) * 100
            
            # Get trial labels
            trial_info = {
                'index': int(idx),
                'weight': float(weight),
                'weight_pct': float(weight_pct),
                'reward': int(dataset.trial_reward[idx]),
                'location': int(dataset.trial_location[idx]),
            }
            
            if hasattr(dataset, 'trial_identity'):
                trial_info['identity'] = int(dataset.trial_identity[idx])
            if hasattr(dataset, 'trial_probability'):
                trial_info['probability'] = int(dataset.trial_probability[idx])
            
            results['outlier_details'].append(trial_info)
            
            print(f"Trial {idx}: weight={weight:.2f} ({weight_pct:.1f}%)")
            print(f"  Reward: {trial_info['reward']}, Location: {trial_info['location']}")
            if 'identity' in trial_info:
                print(f"  Identity: {trial_info['identity']}")
        
        # Analyze target trial activity patterns
        print(f"\n{'='*60}")
        print("OUTLIER vs NON-OUTLIER ACTIVITY COMPARISON")
        print(f"{'='*60}")
        
        outlier_activity = target_pop[outlier_indices].cpu().numpy()
        non_outlier_activity = target_pop[~torch.tensor(outlier_mask)].cpu().numpy()
        
        print(f"Outlier trials mean activity: {outlier_activity.mean():.2f}")
        print(f"Non-outlier trials mean activity: {non_outlier_activity.mean():.2f}")
        print(f"Outlier trials temporal std: {outlier_activity.std(axis=1).mean():.3f}")
        print(f"Non-outlier trials temporal std: {non_outlier_activity.std(axis=1).mean():.3f}")
        
        results['outlier_mean_activity'] = float(outlier_activity.mean())
        results['non_outlier_mean_activity'] = float(non_outlier_activity.mean())
        
        # Plot weight distribution
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        ax = axes[0, 0]
        ax.hist(target_weight_sum, bins=50, edgecolor='black')
        ax.axvline(mean_weight + 2*std_weight, color='r', linestyle='--', label='2 std threshold')
        ax.set_xlabel('Weight sum per target trial')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Matching Weights')
        ax.legend()
        
        ax = axes[0, 1]
        ax.bar(range(len(target_weight_sum)), target_weight_sum, alpha=0.7)
        ax.scatter(outlier_indices, target_weight_sum[outlier_mask], color='red', s=50, label='Outliers')
        ax.axhline(mean_weight + 2*std_weight, color='r', linestyle='--')
        ax.set_xlabel('Target trial index')
        ax.set_ylabel('Weight sum')
        ax.set_title('Matching Weights by Trial')
        ax.legend()
        
        ax = axes[1, 0]
        if len(outlier_indices) > 0:
            for i, idx in enumerate(outlier_indices[:5]):
                ax.plot(target_pop[idx].cpu().numpy(), label=f'Trial {idx}', alpha=0.7)
            ax.set_xlabel('Time bin')
            ax.set_ylabel('Population activity')
            ax.set_title('Outlier Trial Activity Patterns')
            ax.legend()
        
        ax = axes[1, 1]
        im = ax.imshow(soft_weights.cpu().numpy(), aspect='auto', cmap='hot')
        ax.set_xlabel('Target trial')
        ax.set_ylabel('Model trial')
        ax.set_title('Soft Assignment Matrix')
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'matching_analysis.png', dpi=150)
        plt.close()
        
        # Save results
        with open(output_dir / 'analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_dir}")
        
        return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze trial matching concentration')
    parser.add_argument('--data', type=str, required=True, help='Path to .mat file')
    parser.add_argument('--model', type=str, default=None, help='Path to trained model checkpoint')
    parser.add_argument('--output', type=str, default='results/target2_analysis', help='Output directory')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    
    args = parser.parse_args()
    
    # Load data
    dataset = load_session(args.data, validate=False)
    train_idx, val_idx = train_val_split(dataset, val_fraction=0.2, seed=42)
    
    all_data = dataset.get_all_trials()
    train_data = {k: all_data[k][train_idx] for k in ['inputs', 'targets', 'mask']}
    
    # Create or load model
    neuron_info = dataset.get_neuron_info()
    model = create_model_from_data(
        n_classic=neuron_info['n_exc'],
        n_interneuron=neuron_info['n_inh'],
        n_inputs=dataset.get_input_dim(),
        dt=float(dataset.bin_size_ms),
        device=args.device
    )
    
    if args.model:
        checkpoint = torch.load(args.model, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from: {args.model}")
    else:
        print("Using untrained model (random initialization)")
    
    # Create a subset dataset for the training indices
    class SubsetDataset:
        def __init__(self, full_dataset, indices):
            self.trial_reward = full_dataset.trial_reward[indices]
            self.trial_location = full_dataset.trial_location[indices]
            if hasattr(full_dataset, 'trial_identity'):
                self.trial_identity = full_dataset.trial_identity[indices]
            if hasattr(full_dataset, 'trial_probability'):
                self.trial_probability = full_dataset.trial_probability[indices]
    
    subset = SubsetDataset(dataset, train_idx)
    analyze_matching_concentration(model, train_data, subset, args.device, args.output)


if __name__ == "__main__":
    main()
