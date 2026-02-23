"""
Validation loop for age-gender multi-task model.

Provides function to evaluate model on validation set during training.
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict


def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn,
    device: torch.device,
    verbose: bool = True
) -> Dict[str, float]:
    """Validate model on validation set.
    
    Args:
        model: Multi-task neural network
        dataloader: Validation data loader
        loss_fn: MultiTaskLoss instance
        device: Device to evaluate on (cuda or cpu)
        verbose: Whether to show progress bar
        
    Returns:
        Dictionary with average validation losses:
            - 'total': Average total loss
            - 'age': Average age loss
            - 'gender': Average gender loss
    """
    model.eval()
    
    running_losses = {
        'total': 0.0,
        'age': 0.0,
        'gender': 0.0
    }
    
    num_batches = len(dataloader)
    
    # Progress bar
    if verbose:
        pbar = tqdm(dataloader, desc="Validation")
    else:
        pbar = dataloader
    
    with torch.no_grad():
        for images, ages, genders in pbar:
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
            
            # Accumulate losses
            running_losses['total'] += loss_dict['total']
            running_losses['age'] += loss_dict['age']
            running_losses['gender'] += loss_dict['gender']
            
            # Update progress bar
            if verbose:
                pbar.set_postfix({
                    'val_loss': f"{loss_dict['total']:.4f}",
                    'age': f"{loss_dict['age']:.4f}",
                    'gender': f"{loss_dict['gender']:.4f}"
                })
    
    # Calculate average losses
    avg_losses = {
        key: value / num_batches
        for key, value in running_losses.items()
    }
    
    return avg_losses
