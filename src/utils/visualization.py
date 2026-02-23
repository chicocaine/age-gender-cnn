"""
Visualization utilities for model evaluation and analysis.

Provides plotting functions for training curves, confusion matrices,
and prediction analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: tuple = (15, 5)
):
    """Plot training and validation loss curves.
    
    Args:
        history: Dictionary with keys like 'train_total', 'val_total', 'train_age', etc.
        save_path: Optional path to save figure
        figsize: Figure size (width, height)
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    epochs = range(1, len(history['train_total']) + 1)
    
    # Total loss
    axes[0].plot(epochs, history['train_total'], 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_total'], 'r-', label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Total Loss', fontsize=12)
    axes[0].set_title('Combined Loss', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Age loss
    axes[1].plot(epochs, history['train_age'], 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, history['val_age'], 'r-', label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Age MAE', fontsize=12)
    axes[1].set_title('Age Regression Loss', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Gender loss
    axes[2].plot(epochs, history['train_gender'], 'b-', label='Train', linewidth=2)
    axes[2].plot(epochs, history['val_gender'], 'r-', label='Validation', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Gender BCE', fontsize=12)
    axes[2].set_title('Gender Classification Loss', fontsize=14)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training history plot to {save_path}")
    
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str] = ['Male', 'Female'],
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6)
):
    """Plot confusion matrix as heatmap.
    
    Args:
        cm: Confusion matrix (2x2 for gender)
        labels: Class labels
        save_path: Optional path to save figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Normalize to percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotations with both count and percentage
    annotations = np.array([[f'{count}\n({percent:.1f}%)' 
                            for count, percent in zip(row_counts, row_percents)]
                           for row_counts, row_percents in zip(cm, cm_percent)])
    
    sns.heatmap(
        cm,
        annot=annotations,
        fmt='',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'},
        square=True
    )
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title('Gender Classification Confusion Matrix', fontsize=14)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    
    plt.show()


def plot_age_error_distribution(
    errors: np.ndarray,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6)
):
    """Plot histogram of age prediction errors.
    
    Args:
        errors: Array of prediction errors (predicted - actual)
        save_path: Optional path to save figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    plt.hist(errors, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    plt.axvline(np.mean(errors), color='green', linestyle='--', linewidth=2, 
                label=f'Mean Error: {np.mean(errors):.2f}')
    
    plt.xlabel('Prediction Error (years)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Age Prediction Error Distribution', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'MAE: {np.mean(np.abs(errors)):.2f}\nStd: {np.std(errors):.2f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved error distribution to {save_path}")
    
    plt.show()


def plot_predictions_vs_actual(
    predictions: np.ndarray,
    actuals: np.ndarray,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 10)
):
    """Plot predicted vs actual ages with identity line.
    
    Args:
        predictions: Predicted ages
        actuals: Ground truth ages
        save_path: Optional path to save figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Calculate errors for color coding
    errors = np.abs(predictions - actuals)
    
    # Scatter plot with error-based coloring
    scatter = plt.scatter(actuals, predictions, c=errors, cmap='YlOrRd',
                         alpha=0.5, s=20, edgecolors='none')
    plt.colorbar(scatter, label='Absolute Error (years)')
    
    # Identity line (perfect predictions)
    min_age = min(actuals.min(), predictions.min())
    max_age = max(actuals.max(), predictions.max())
    plt.plot([min_age, max_age], [min_age, max_age], 'k--', linewidth=2, 
             label='Perfect Prediction', alpha=0.7)
    
    plt.xlabel('Actual Age (years)', fontsize=12)
    plt.ylabel('Predicted Age (years)', fontsize=12)
    plt.title('Age Predictions vs Ground Truth', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Calculate and display MAE
    mae = np.mean(errors)
    plt.text(0.05, 0.95, f'MAE: {mae:.2f} years', transform=plt.gca().transAxes,
             verticalalignment='top', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved predictions vs actual plot to {save_path}")
    
    plt.show()


def visualize_sample_predictions(
    images: np.ndarray,
    true_ages: np.ndarray,
    true_genders: np.ndarray,
    pred_ages: np.ndarray,
    pred_genders: np.ndarray,
    num_samples: int = 10,
    save_path: Optional[str] = None,
    denormalize: bool = True
):
    """Visualize sample predictions with ground truth labels.
    
    Args:
        images: Array of images (N, H, W, 3) or (N, 3, H, W)
        true_ages: Ground truth ages
        true_genders: Ground truth gender labels (0 or 1)
        pred_ages: Predicted ages
        pred_genders: Predicted gender probabilities
        num_samples: Number of samples to display
        save_path: Optional path to save figure
        denormalize: Whether to denormalize images (if ImageNet normalized)
    """
    num_samples = min(num_samples, len(images))
    cols = 5
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    gender_labels = {0: 'M', 1: 'F'}
    
    for idx in range(num_samples):
        # Handle image format
        img = images[idx]
        if img.shape[0] == 3:  # (3, H, W) -> (H, W, 3)
            img = np.transpose(img, (1, 2, 0))
        
        # Denormalize if needed
        if denormalize:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
        
        axes[idx].imshow(img)
        axes[idx].axis('off')
        
        # Create label
        true_gender_label = gender_labels[int(true_genders[idx])]
        pred_gender_label = gender_labels[int(pred_genders[idx] > 0.5)]
        
        error = abs(pred_ages[idx] - true_ages[idx])
        color = 'green' if error < 5 else ('orange' if error < 10 else 'red')
        
        title = f"True: {true_ages[idx]:.0f}y, {true_gender_label}\n"
        title += f"Pred: {pred_ages[idx]:.0f}y, {pred_gender_label}"
        
        axes[idx].set_title(title, fontsize=10, color=color)
    
    # Hide unused subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved sample predictions to {save_path}")
    
    plt.show()


def plot_metrics_by_age_range(
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6)
):
    """Plot MAE across different age ranges.
    
    Args:
        metrics_dict: Dictionary from calculate_metrics_by_age_range
        save_path: Optional path to save figure
        figsize: Figure size
    """
    age_ranges = list(metrics_dict.keys())
    maes = [metrics_dict[r]['mae'] for r in age_ranges]
    counts = [metrics_dict[r]['count'] for r in age_ranges]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # MAE by age range
    bars1 = ax1.bar(age_ranges, maes, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Age Range', fontsize=12)
    ax1.set_ylabel('Mean Absolute Error (years)', fontsize=12)
    ax1.set_title('Age Prediction Error by Age Range', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mae in zip(bars1, maes):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{mae:.2f}', ha='center', va='bottom')
    
    # Sample counts by age range
    bars2 = ax2.bar(age_ranges, counts, color='coral', alpha=0.7)
    ax2.set_xlabel('Age Range', fontsize=12)
    ax2.set_ylabel('Sample Count', fontsize=12)
    ax2.set_title('Dataset Distribution by Age Range', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, count in zip(bars2, counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved age range metrics to {save_path}")
    
    plt.show()
