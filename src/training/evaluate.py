"""
Model evaluation on test sets.

Provides comprehensive evaluation with metrics computation.
Includes both UTKFace (continuous age) and Adience (age-bin) evaluation.
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Evaluate model on test set and compute metrics.
    
    Args:
        model: Trained multi-task neural network
        dataloader: Test data loader
        device: Device to evaluate on (cuda or cpu)
        verbose: Whether to show progress bar
        
    Returns:
        Tuple of (age_predictions, gender_predictions, age_targets, gender_targets, metrics) where:
            - age_predictions: Array of predicted ages
            - gender_predictions: Array of predicted gender probabilities
            - age_targets: Array of ground truth ages
            - gender_targets: Array of ground truth gender labels
            - metrics: Dictionary with 'age_mae' and 'gender_accuracy'
    """
    model.eval()
    
    age_preds = []
    gender_preds = []
    age_targets = []
    gender_targets = []
    
    if verbose:
        pbar = tqdm(dataloader, desc="Evaluating")
    else:
        pbar = dataloader
    
    with torch.no_grad():
        for images, ages, genders in pbar:
            # Move data to device
            images = images.to(device)
            
            # Forward pass
            age_pred, gender_pred = model(images)
            
            # Collect predictions and targets
            age_preds.append(age_pred.cpu().numpy())
            gender_preds.append(gender_pred.cpu().numpy())
            age_targets.append(ages.numpy())
            gender_targets.append(genders.numpy())
    
    # Concatenate all batches
    age_preds = np.concatenate(age_preds, axis=0).squeeze()
    gender_preds = np.concatenate(gender_preds, axis=0).squeeze()
    age_targets = np.concatenate(age_targets, axis=0).squeeze()
    gender_targets = np.concatenate(gender_targets, axis=0).squeeze()
    
    # Compute metrics
    age_mae = np.mean(np.abs(age_preds - age_targets))
    
    # Apply threshold for gender accuracy
    gender_pred_labels = (gender_preds > 0.5).astype(int)
    gender_accuracy = np.mean(gender_pred_labels == gender_targets)
    
    metrics = {
        'age_mae': float(age_mae),
        'gender_accuracy': float(gender_accuracy)
    }
    
    if verbose:
        print(f"\nEvaluation Results:")
        print(f"  Age MAE: {age_mae:.2f} years")
        print(f"  Gender Accuracy: {gender_accuracy:.4f} ({gender_accuracy*100:.2f}%)")
    
    return age_preds, gender_preds, age_targets, gender_targets, metrics


def evaluate_adience(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """Evaluate model on the Adience cross-dataset benchmark.

    The Adience dataloader yields ``(image, age_bin_str, gender_int)`` rather
    than continuous age targets, so this function returns the raw predicted
    continuous ages alongside the ground-truth bin labels.  All bin-level
    metrics (within-range accuracy, ±1-bin tolerance, per-bin breakdown) are
    computed in the notebook for full transparency.

    Args:
        model:      Trained AgeGenderModel in eval mode.
        dataloader: AdienceDataset DataLoader using ``adience_collate_fn``.
        device:     Torch device (CPU or CUDA).
        verbose:    Show tqdm progress bar and summary statistics.

    Returns:
        Tuple of:
            age_preds    – continuous predicted ages (N,)
            gender_preds – raw sigmoid outputs (N,)
            true_bins    – ground-truth Adience bin label strings, length N
            true_genders – ground-truth gender integers 0/1 (N,)
    """
    model.eval()

    age_preds_list:    list = []
    gender_preds_list: list = []
    true_bins_list:    List[str] = []
    true_genders_list: list = []

    pbar = tqdm(dataloader, desc="Evaluating Adience") if verbose else dataloader

    with torch.no_grad():
        for batch in pbar:
            if batch is None:
                continue
            images, age_bins, genders = batch
            images = images.to(device)

            age_pred, gender_pred = model(images)

            age_preds_list.append(age_pred.cpu().numpy())
            gender_preds_list.append(gender_pred.cpu().numpy())
            true_bins_list.extend(age_bins)
            true_genders_list.append(genders.numpy())

    age_preds    = np.concatenate(age_preds_list,    axis=0).squeeze()
    gender_preds = np.concatenate(gender_preds_list, axis=0).squeeze()
    true_genders = np.concatenate(true_genders_list, axis=0).squeeze()

    if verbose:
        gender_acc = np.mean((gender_preds > 0.5).astype(int) == true_genders)
        print(f"\nAdience Evaluation — {len(age_preds)} samples")
        print(f"  Mean predicted age : {age_preds.mean():.1f} years")
        print(f"  Gender accuracy    : {gender_acc:.4f} ({gender_acc*100:.2f}%)")

    return age_preds, gender_preds, true_bins_list, true_genders
