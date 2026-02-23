"""
Configuration management for experiments.

Provides dataclass-based configuration and JSON serialization for
reproducible experiments.
"""

import json
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Configuration for model training experiment.
    
    Captures all hyperparameters and settings for reproducibility.
    """
    # Experiment metadata
    experiment_id: str = "baseline"
    description: str = "Baseline multi-task age-gender model"
    
    # Model architecture
    model_architecture: str = "MobileNetV2"
    pretrained_backbone: bool = True
    freeze_backbone: bool = False
    dropout_rate: float = 0.2
    
    # Training hyperparameters
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 20
    optimizer: str = "Adam"
    
    # Loss function
    age_loss_weight: float = 1.0
    gender_loss_weight: float = 1.0
    
    # Data settings
    image_size: tuple = (224, 224)
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Augmentation strategy  
    augmentation: str = "full"  # Options: "full", "minimal", "none"
    
    # Reproducibility
    random_seed: int = 42
    
    # Paths
    data_dir: str = "dataset/raw/UTKFace"
    metadata_path: str = "dataset/processed/utkface_metadata.csv"
    checkpoint_dir: str = "models"
    experiment_dir: str = "experiments"
    
    # Device
    device: str = "cuda"  # or "cpu"
    
    # Training options
    save_best_only: bool = True
    early_stopping_patience: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def to_json(self, filepath: str):
        """Save config to JSON file.
        
        Args:
            filepath: Path to save JSON file
        """
        config_dict = self.to_dict()
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Saved config to {filepath}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Load config from dictionary.
        
        Args:
            config_dict: Dictionary with config parameters
            
        Returns:
            ExperimentConfig instance
        """
        # Handle tuple conversion (JSON stores as list)
        if 'image_size' in config_dict and isinstance(config_dict['image_size'], list):
            config_dict['image_size'] = tuple(config_dict['image_size'])
        
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, filepath: str) -> 'ExperimentConfig':
        """Load config from JSON file.
        
        Args:
            filepath: Path to JSON config file
            
        Returns:
            ExperimentConfig instance
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)


def get_default_config() -> ExperimentConfig:
    """Get default experiment configuration.
    
    Returns:
        ExperimentConfig with default values
    """
    return ExperimentConfig()


def create_experiment_config(
    experiment_id: str,
    description: str,
    **kwargs
) -> ExperimentConfig:
    """Create custom experiment configuration.
    
    Args:
        experiment_id: Unique experiment identifier
        description: Experiment description
        **kwargs: Additional config parameters to override defaults
        
    Returns:
        ExperimentConfig instance
    """
    config = get_default_config()
    config.experiment_id = experiment_id
    config.description = description
    
    # Override with provided kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown config parameter '{key}' ignored")
    
    return config


def save_experiment_results(
    config: ExperimentConfig,
    metrics: Dict[str, float],
    history: Dict[str, list],
    filepath: str
):
    """Save complete experiment results (config + metrics + history).
    
    Args:
        config: Experiment configuration
        metrics: Final evaluation metrics
        history: Training history (loss curves)
        filepath: Path to save results JSON
    """
    results = {
        'config': config.to_dict(),
        'metrics': metrics,
        'history': history
    }
    
    # Ensure directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved experiment results to {filepath}")


def load_experiment_results(filepath: str) -> Dict[str, Any]:
    """Load experiment results from JSON.
    
    Args:
        filepath: Path to results JSON
        
    Returns:
        Dictionary with config, metrics, and history
    """
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    # Reconstruct config object
    results['config'] = ExperimentConfig.from_dict(results['config'])
    
    return results
