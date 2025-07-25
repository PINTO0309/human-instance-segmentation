"""Text file logging utility for training and validation metrics in a single file."""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


class TextLogger:
    """Logger that writes training and validation metrics to a single text file."""
    
    def __init__(self, log_dir: Path):
        """Initialize text logger.
        
        Args:
            log_dir: Directory to save log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create single log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = self.log_dir / f"training_log_{timestamp}.txt"
        
        # Write header
        self._write_header()
        
    def _write_header(self):
        """Write header to log file."""
        with open(self.log_path, 'w') as f:
            f.write(f"{'='*60}\n")
            f.write(f"Training and Validation Log\n")
            f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*60}\n\n")
    
    
    def log_epoch_summary(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
        learning_rate: float = 0.0
    ):
        """Log epoch summary to both files.
        
        Args:
            epoch: Current epoch number
            train_metrics: Training metrics for the epoch
            val_metrics: Validation metrics for the epoch
            learning_rate: Current learning rate
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(self.log_path, 'a') as f:
            f.write(f"{'='*40}\n")
            f.write(f"[{timestamp}] Epoch {epoch+1} Summary\n")
            f.write(f"Learning Rate: {learning_rate:.6f}\n")
            
            # Log training metrics
            f.write("Training Metrics:\n")
            for key, value in sorted(train_metrics.items()):
                if isinstance(value, (int, float)):
                    f.write(f"  {key}: {value:.4f}\n")
            
            # Log validation metrics if available
            if val_metrics is not None:
                f.write("\nValidation Metrics:\n")
                for key, value in sorted(val_metrics.items()):
                    if isinstance(value, (int, float)):
                        f.write(f"  {key}: {value:.4f}\n")
            
            f.write(f"{'='*40}\n\n")
    
    
    def log_best_model(
        self,
        epoch: int,
        miou: float,
        checkpoint_path: str
    ):
        """Log when a new best model is saved.
        
        Args:
            epoch: Epoch number
            miou: Best mIoU value
            checkpoint_path: Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(self.log_path, 'a') as f:
            f.write(f"{'*'*60}\n")
            f.write(f"[{timestamp}] NEW BEST MODEL\n")
            f.write(f"Epoch: {epoch+1}\n")
            f.write(f"mIoU: {miou:.4f}\n")
            f.write(f"Saved to: {checkpoint_path}\n")
            f.write(f"{'*'*60}\n\n")
    
    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration.
        
        Args:
            config: Configuration dictionary
        """
        config_str = json.dumps(config, indent=2)
        
        with open(self.log_path, 'a') as f:
            f.write("Experiment Configuration:\n")
            f.write(config_str)
            f.write("\n\n")
    
    def log_error(self, error_msg: str):
        """Log error messages to the log file.
        
        Args:
            error_msg: Error message to log
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        error_str = f"[{timestamp}] ERROR: {error_msg}\n\n"
        
        with open(self.log_path, 'a') as f:
            f.write(error_str)
    
    def close(self):
        """Write closing message to log file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        closing_msg = f"\n{'='*60}\nTraining completed at: {timestamp}\n{'='*60}\n"
        
        with open(self.log_path, 'a') as f:
            f.write(closing_msg)