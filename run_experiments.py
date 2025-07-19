"""Run and compare experiments with different configurations."""

import argparse
import json
from pathlib import Path
from typing import List, Dict
import subprocess
import time
import pandas as pd
import matplotlib.pyplot as plt


def run_experiment(config_name: str, additional_args: List[str] = None) -> Dict:
    """Run a single experiment with given configuration.
    
    Args:
        config_name: Configuration name
        additional_args: Additional command line arguments
        
    Returns:
        Dictionary with experiment results
    """
    print(f"\n{'='*60}")
    print(f"Running experiment: {config_name}")
    print(f"{'='*60}")
    
    # Build command
    cmd = [
        'python', 'train_advanced.py',
        '--config', config_name
    ]
    
    if additional_args:
        cmd.extend(additional_args)
        
    # Record start time
    start_time = time.time()
    
    # Run experiment
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Experiment failed with error:")
            print(result.stderr)
            return {
                'config': config_name,
                'status': 'failed',
                'error': result.stderr
            }
            
        # Parse results from output or load from saved metrics
        end_time = time.time()
        
        return {
            'config': config_name,
            'status': 'completed',
            'duration': end_time - start_time,
            'output': result.stdout
        }
        
    except Exception as e:
        return {
            'config': config_name,
            'status': 'error',
            'error': str(e)
        }


def load_experiment_metrics(experiment_dir: Path) -> Dict:
    """Load metrics from completed experiment.
    
    Args:
        experiment_dir: Path to experiment directory
        
    Returns:
        Dictionary with metrics
    """
    metrics = {}
    
    # Load best checkpoint metrics
    best_checkpoint = experiment_dir / 'checkpoints' / 'best_model.pth'
    if best_checkpoint.exists():
        import torch
        checkpoint = torch.load(best_checkpoint, map_location='cpu')
        metrics['best_miou'] = checkpoint.get('best_miou', 0)
        metrics['best_epoch'] = checkpoint.get('epoch', 0)
        
    # Load validation history from logs
    # This would require parsing TensorBoard logs or saving metrics separately
    
    return metrics


def compare_experiments(experiment_names: List[str], output_dir: str = 'experiments') -> pd.DataFrame:
    """Compare metrics across experiments.
    
    Args:
        experiment_names: List of experiment names
        output_dir: Base output directory
        
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for exp_name in experiment_names:
        exp_dir = Path(output_dir) / exp_name
        
        if not exp_dir.exists():
            print(f"Warning: Experiment directory not found: {exp_dir}")
            continue
            
        # Load config
        config_path = exp_dir / 'configs' / 'config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {}
            
        # Load metrics
        metrics = load_experiment_metrics(exp_dir)
        
        # Compile results
        result = {
            'experiment': exp_name,
            'multiscale': config.get('multiscale', {}).get('enabled', False),
            'distance_loss': config.get('distance_loss', {}).get('enabled', False),
            'cascade': config.get('cascade', {}).get('enabled', False),
            'best_miou': metrics.get('best_miou', 0),
            'best_epoch': metrics.get('best_epoch', 0)
        }
        
        results.append(result)
        
    return pd.DataFrame(results)


def plot_experiment_comparison(df: pd.DataFrame, save_path: str = 'experiment_comparison.png'):
    """Plot comparison of experiments.
    
    Args:
        df: DataFrame with experiment results
        save_path: Path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: mIoU comparison
    ax1.bar(df['experiment'], df['best_miou'])
    ax1.set_xlabel('Experiment')
    ax1.set_ylabel('Best mIoU')
    ax1.set_title('Best mIoU Comparison')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Feature impact
    features = ['multiscale', 'distance_loss', 'cascade']
    feature_impact = {}
    
    for feature in features:
        with_feature = df[df[feature] == True]['best_miou'].mean()
        without_feature = df[df[feature] == False]['best_miou'].mean()
        feature_impact[feature] = with_feature - without_feature
        
    ax2.bar(feature_impact.keys(), feature_impact.values())
    ax2.set_xlabel('Feature')
    ax2.set_ylabel('mIoU Impact')
    ax2.set_title('Feature Impact on mIoU')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved comparison plot to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Run and compare experiments')
    
    parser.add_argument('--configs', nargs='+', default=['baseline', 'multiscale', 'multiscale_distance', 'full'],
                        help='List of configurations to run')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs per experiment')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training and only compare existing results')
    parser.add_argument('--output_dir', type=str, default='experiments',
                        help='Output directory for experiments')
    
    args = parser.parse_args()
    
    if not args.skip_training:
        # Run experiments
        results = []
        
        for config in args.configs:
            # Prepare additional arguments
            additional_args = [
                '--config_modifications',
                json.dumps({
                    'training.num_epochs': args.epochs,
                    'training.batch_size': args.batch_size
                })
            ]
            
            result = run_experiment(config, additional_args)
            results.append(result)
            
            # Save intermediate results
            with open('experiment_runs.json', 'w') as f:
                json.dump(results, f, indent=2)
                
        print("\n" + "="*60)
        print("All experiments completed!")
        print("="*60)
        
    # Compare results
    print("\nComparing experiment results...")
    
    df = compare_experiments(args.configs, args.output_dir)
    
    if not df.empty:
        print("\nExperiment Comparison:")
        print(df.to_string(index=False))
        
        # Save comparison
        df.to_csv('experiment_comparison.csv', index=False)
        
        # Plot results
        plot_experiment_comparison(df)
        
        # Print summary
        print("\nSummary:")
        print(f"Best performing config: {df.loc[df['best_miou'].idxmax(), 'experiment']}")
        print(f"Best mIoU achieved: {df['best_miou'].max():.4f}")
        
        # Feature impact analysis
        print("\nFeature Impact Analysis:")
        for feature in ['multiscale', 'distance_loss', 'cascade']:
            with_feature = df[df[feature] == True]['best_miou'].mean()
            without_feature = df[df[feature] == False]['best_miou'].mean()
            impact = with_feature - without_feature
            print(f"  {feature}: {impact:+.4f} mIoU")


if __name__ == "__main__":
    main()