"""
Training loop for age-gender multi-task model.

Provides function to train model for one epoch with loss tracking.
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Optional


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    device: torch.device,
    epoch: int = 0,
    verbose: bool = True
) -> Dict[str, float]:
    """Train model for one epoch.
    
    Args:
        model: Multi-task neural network
        dataloader: Training data loader
        optimizer: PyTorch optimizer
        loss_fn: MultiTaskLoss instance
        device: Device to train on (cuda or cpu)
        epoch: Current epoch number (for display)
        verbose: Whether to show progress bar
        
    Returns:
        Dictionary with average losses:
            - 'total': Average total loss
            - 'age': Average age loss
            - 'gender': Average gender loss
    """
    model.train()
    
    running_losses = {
        'total': 0.0,
        'age': 0.0,
        'gender': 0.0
    }
    
    num_batches = len(dataloader)
    
    # Progress bar
    if verbose:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    else:
        pbar = dataloader
    
    for batch_idx, (images, ages, genders) in enumerate(pbar):
        # Move data to device
        images = images.to(device)
        ages = ages.to(device)
        genders = genders.to(device)
        
        # Forward pass
        age_pred, gender_pred = model(images)
        
        # Compute loss
        loss, loss_dict = loss_fn(
            age_pred.squeeze(),
            gender_pred.squeeze(),
            ages.squeeze(),
            genders.squeeze()
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        running_losses['total'] += loss_dict['total']
        running_losses['age'] += loss_dict['age']
        running_losses['gender'] += loss_dict['gender']
        
        # Update progress bar
        if verbose:
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'age': f"{loss_dict['age']:.4f}",
                'gender': f"{loss_dict['gender']:.4f}"
            })
    
    # Calculate average losses
    avg_losses = {
        key: value / num_batches
        for key, value in running_losses.items()
    }
    
    return avg_losses
