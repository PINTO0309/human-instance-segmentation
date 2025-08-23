"""Run and compare experiments with different configurations."""

import argparse
import json
from pathlib import Path
from typing import List, Dict
import subprocess
import time
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch

# Set environment variable for tqdm to use dynamic ncols
os.environ['TQDM_DYNAMIC_NCOLS'] = 'True'


def export_untrained_model_to_onnx(config_name: str, output_dir: str = 'experiments') -> bool:
    """Create and export an untrained model to ONNX format.

    Args:
        config_name: Name of the configuration
        output_dir: Base output directory

    Returns:
        Success status
    """
    from src.human_edge_detection.experiments.config_manager import ConfigManager, ExperimentConfig, create_experiment_dirs
    from src.human_edge_detection.export_onnx_advanced import export_checkpoint_to_onnx_advanced

    # Load configuration
    if config_name in ConfigManager.list_configs():
        config = ConfigManager.get_config(config_name)
    else:
        return False

    # Create experiment directories
    exp_dirs = create_experiment_dirs(config)

    # Import model creation functions
    from train_advanced import build_model

    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Build model
    print(f"\nBuilding untrained {config_name} model...")
    model, feature_extractor = build_model(config, device)

    # Save untrained model
    untrained_checkpoint_path = exp_dirs['checkpoints'] / 'untrained_model.pth'
    print(f"Saving untrained model to {untrained_checkpoint_path}")
    torch.save({
        'epoch': -1,  # -1 indicates untrained
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': {},  # Empty optimizer state
        'scheduler_state_dict': None,
        'best_miou': 0.0,
        'config': config.to_dict()
    }, untrained_checkpoint_path)

    # Export to ONNX
    untrained_onnx_path = exp_dirs['checkpoints'] / 'untrained_model.onnx'
    
    # Determine model type
    if config.model.use_hierarchical or any(getattr(config.model, attr, False) for attr in ['use_hierarchical_unet', 'use_hierarchical_unet_v2', 'use_hierarchical_unet_v3', 'use_hierarchical_unet_v4']):
        model_type = 'hierarchical'
    elif config.model.use_class_specific_decoder:
        model_type = 'class_specific'
    elif config.multiscale.enabled:
        model_type = 'multiscale'
    else:
        model_type = 'baseline'

    print(f"Exporting untrained model to ONNX...")
    success = export_checkpoint_to_onnx_advanced(
        checkpoint_path=str(untrained_checkpoint_path),
        output_path=str(untrained_onnx_path),
        model_type=model_type,
        config=config.to_dict(),
        device=device,
        verify=False  # Skip verification for untrained models
    )

    if success:
        print(f"Successfully exported untrained model to: {untrained_onnx_path}")
    else:
        print(f"Failed to export untrained model for {config_name}")

    return success


def export_model_to_onnx(experiment_name: str, output_dir: str = 'experiments', checkpoint_name: str = 'best_model.pth') -> bool:
    """Export a model from an experiment to ONNX format.

    Args:
        experiment_name: Name of the experiment
        output_dir: Base output directory
        checkpoint_name: Name of checkpoint file to export

    Returns:
        Success status
    """
    exp_dir = Path(output_dir) / experiment_name
    checkpoint_path = exp_dir / 'checkpoints' / checkpoint_name

    if not checkpoint_path.exists():
        # No model to export
        print(f"\nNo checkpoint found for {experiment_name} at {checkpoint_path}")
        print("  → Skipping ONNX export (no model to export)")
        return False

    # Load config from experiment
    config_path = exp_dir / 'configs' / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            exp_config = json.load(f)
    else:
        print(f"Warning: Config not found for {experiment_name}")
        return False

    # Determine model type
    is_multiscale = exp_config.get('multiscale', {}).get('enabled', False)
    is_hierarchical = exp_config.get('model', {}).get('use_hierarchical', False)
    is_class_specific = exp_config.get('model', {}).get('use_class_specific_decoder', False)
    
    # Check for all hierarchical model variants
    model_config = exp_config.get('model', {})
    is_any_hierarchical = (
        is_hierarchical or
        model_config.get('use_hierarchical_unet', False) or
        model_config.get('use_hierarchical_unet_v2', False) or
        model_config.get('use_hierarchical_unet_v3', False) or
        model_config.get('use_hierarchical_unet_v4', False) or
        model_config.get('use_rgb_hierarchical', False)
    )
    
    if is_any_hierarchical:
        model_type = 'hierarchical'
    elif is_class_specific:
        model_type = 'class_specific'
    elif is_multiscale:
        model_type = 'multiscale'
    else:
        model_type = 'baseline'

    # Import advanced export function
    from src.human_edge_detection.export_onnx_advanced import export_checkpoint_to_onnx_advanced

    # Export path
    onnx_name = checkpoint_name.replace('.pth', '.onnx')
    onnx_path = exp_dir / 'checkpoints' / onnx_name

    print(f"\nExporting {model_type} model for {experiment_name}...")
    print(f"Checkpoint: {checkpoint_name}")

    success = export_checkpoint_to_onnx_advanced(
        checkpoint_path=str(checkpoint_path),
        output_path=str(onnx_path),
        model_type=model_type,
        config=exp_config,  # Already a dict
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verify=True
    )

    if success:
        print(f"ONNX model exported to: {onnx_path}")
    else:
        print(f"Failed to export ONNX model for {experiment_name}")
        print("Note: Multiscale models with complex architectures may require custom export logic.")

    return success


def run_experiment(config_name: str, additional_args: List[str] = None, resume_checkpoint: str = None,
                  teacher_checkpoint: str = None, distillation_params: dict = None, mixed_precision: bool = False) -> Dict:
    """Run a single experiment with given configuration.

    Args:
        config_name: Configuration name
        additional_args: Additional command line arguments
        resume_checkpoint: Path to checkpoint to resume from
        teacher_checkpoint: Path to teacher model checkpoint for distillation
        distillation_params: Dictionary with distillation parameters (temperature, alpha)

    Returns:
        Dictionary with experiment results
    """
    print(f"\n{'='*60}")
    print(f"Running experiment: {config_name}")
    if resume_checkpoint:
        print(f"Resuming from: {resume_checkpoint}")
    if teacher_checkpoint:
        print(f"Using teacher checkpoint: {teacher_checkpoint}")
    print(f"{'='*60}")

    # Build command
    cmd = [
        'uv', 'run', 'python', 'train_advanced.py',
        '--config', config_name
    ]

    if resume_checkpoint:
        cmd.extend(['--resume', resume_checkpoint])
    
    # Add distillation parameters
    if teacher_checkpoint:
        cmd.extend(['--teacher_checkpoint', teacher_checkpoint])
    
    if distillation_params:
        if 'temperature' in distillation_params:
            cmd.extend(['--distillation_temperature', str(distillation_params['temperature'])])
        if 'alpha' in distillation_params:
            cmd.extend(['--distillation_alpha', str(distillation_params['alpha'])])
    
    if mixed_precision:
        cmd.append('--mixed_precision')

    if additional_args:
        cmd.extend(additional_args)

    # Record start time
    start_time = time.time()

    # Run experiment with real-time output
    try:
        # Use Popen with inherited stdout/stderr for proper tqdm display
        # But also capture output using tee for logging
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.log') as temp_log:
            temp_log_path = temp_log.name
        
        # Run with tee to both display and capture output
        # Properly escape the command for shell
        import shlex
        escaped_cmd = ' '.join(shlex.quote(arg) for arg in cmd)
        tee_cmd = f"{escaped_cmd} 2>&1 | tee {temp_log_path}"
        process = subprocess.Popen(
            tee_cmd,
            shell=True
        )

        # Wait for process to complete
        return_code = process.wait()

        # Read captured output
        with open(temp_log_path, 'r') as f:
            full_output = f.read()
        
        # Clean up temp file
        import os
        os.unlink(temp_log_path)

        if return_code != 0:
            print(f"\nExperiment failed with error code: {return_code}")
            print("\n" + "="*60)
            print("ERROR TRACEBACK:")
            print("="*60)
            # Look for traceback in output
            lines = full_output.split('\n')
            traceback_started = False
            error_lines = []

            for line in lines:
                if 'Traceback' in line:
                    traceback_started = True
                if traceback_started:
                    error_lines.append(line)

            if error_lines:
                # Print the traceback
                for line in error_lines:
                    print(line)
            else:
                # If no traceback found, print last 50 lines of output
                print("No traceback found. Last 50 lines of output:")
                print("-"*60)
                for line in lines[-50:]:
                    if line.strip():  # Only print non-empty lines
                        print(line)
            print("="*60)

            return {
                'config': config_name,
                'status': 'failed',
                'error': full_output
            }

        # Parse results from output or load from saved metrics
        end_time = time.time()
        duration = end_time - start_time

        print(f"\nExperiment '{config_name}' completed in {duration/60:.1f} minutes")

        return {
            'config': config_name,
            'status': 'completed',
            'duration': duration,
            'output': full_output
        }

    except Exception as e:
        import traceback
        print(f"\nExperiment failed with exception: {type(e).__name__}")
        print("\n" + "="*60)
        print("EXCEPTION TRACEBACK:")
        print("="*60)
        traceback.print_exc()
        print("="*60)

        return {
            'config': config_name,
            'status': 'error',
            'error': str(e) + '\n\n' + traceback.format_exc()
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
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training and only compare existing results')
    parser.add_argument('--output_dir', type=str, default='experiments',
                        help='Output directory for experiments')
    parser.add_argument('--export_onnx', action='store_true',
                        help='Export ONNX models for existing experiments')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint (.pth) to resume training from')
    parser.add_argument('--additional_epochs', type=int, default=None,
                        help='Number of additional epochs to train (adds to checkpoint epoch)')
    parser.add_argument('--total_epochs', type=int, default=None,
                        help='Total epochs to train to (overrides --epochs)')
    
    # Distillation arguments
    parser.add_argument('--teacher_checkpoint', type=str, default=None,
                        help='Path to teacher model checkpoint for distillation')
    parser.add_argument('--distillation_temperature', type=float, default=None,
                        help='Temperature for distillation (default: from config)')
    parser.add_argument('--distillation_alpha', type=float, default=None,
                        help='Alpha weight for distillation loss (default: from config)')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Enable mixed precision training for faster training and lower memory usage')

    args = parser.parse_args()
    
    # Validate arguments
    if args.resume:
        if not Path(args.resume).exists():
            print(f"Error: Checkpoint file not found: {args.resume}")
            return
        
        # When resuming, only run one config
        if len(args.configs) > 1:
            print("Warning: When resuming, only the first config will be used")
            args.configs = [args.configs[0]]
            
    if args.additional_epochs and args.total_epochs:
        print("Error: Cannot specify both --additional_epochs and --total_epochs")
        return

    # Export ONNX models for existing experiments
    if args.export_onnx:
        if args.epochs == 0:
            # Export untrained models when epochs is 0
            print("Exporting untrained ONNX models (--epochs 0 specified)...")
            for config in args.configs:
                export_untrained_model_to_onnx(config, args.output_dir)
        else:
            # Export existing trained models
            print("Exporting ONNX models for existing experiments...")
            for config in args.configs:
                export_model_to_onnx(config, args.output_dir)
        return

    if not args.skip_training:
        # Run experiments
        results = []

        # Calculate number of epochs and config when resuming
        if args.resume:
            # Load checkpoint to get current epoch and config
            checkpoint = torch.load(args.resume, map_location='cpu')
            current_epoch = checkpoint.get('epoch', -1) + 1  # +1 because epoch is 0-indexed
            
            # Try to get config from checkpoint
            if 'config' in checkpoint:
                checkpoint_config = checkpoint['config']
                config_name = checkpoint_config.get('name', args.configs[0])
                print(f"\nDetected config from checkpoint: {config_name}")
                # Override the config if it was found
                args.configs = [config_name]
            
            if args.additional_epochs or args.total_epochs:
                if args.total_epochs:
                    # Train to total_epochs
                    num_epochs = args.total_epochs
                    print(f"Resuming from epoch {current_epoch}, training to epoch {num_epochs}")
                elif args.additional_epochs:
                    # Add additional epochs
                    num_epochs = current_epoch + args.additional_epochs
                    print(f"Resuming from epoch {current_epoch}, adding {args.additional_epochs} epochs (total: {num_epochs})")
            else:
                # Default: continue to originally planned epochs
                num_epochs = checkpoint_config.get('training', {}).get('num_epochs', args.epochs) if 'config' in checkpoint else args.epochs
                print(f"Resuming from epoch {current_epoch}, continuing to epoch {num_epochs}")
        else:
            num_epochs = args.total_epochs if args.total_epochs else args.epochs

        for config in args.configs:
            # Export untrained model first if specified
            if args.export_onnx and not args.resume:
                print(f"\nExporting untrained model for {config}...")
                export_untrained_model_to_onnx(config, args.output_dir)
            
            # Prepare additional arguments
            additional_args = [
                '--config_modifications',
                json.dumps({
                    'training.num_epochs': num_epochs,
                    'training.batch_size': args.batch_size
                })
            ]

            # Prepare distillation parameters if provided
            distillation_params = {}
            if args.distillation_temperature is not None:
                distillation_params['temperature'] = args.distillation_temperature
            if args.distillation_alpha is not None:
                distillation_params['alpha'] = args.distillation_alpha
            
            result = run_experiment(
                config, 
                additional_args, 
                resume_checkpoint=args.resume,
                teacher_checkpoint=args.teacher_checkpoint,
                distillation_params=distillation_params if distillation_params else None,
                mixed_precision=args.mixed_precision
            )
            results.append(result)

            # Save intermediate results
            with open('experiment_runs.json', 'w') as f:
                json.dump(results, f, indent=2)

            # Export best model to ONNX after experiment completes
            if result['status'] == 'completed':
                print(f"\nExporting best model for {config}...")
                export_model_to_onnx(config, args.output_dir, 'best_model.pth')

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