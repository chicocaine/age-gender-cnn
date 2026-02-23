"""
Loss functions for multi-task age and gender prediction.

Combines age regression loss (MAE) with gender classification loss (BCE).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class MultiTaskLoss(nn.Module):
    """Combined loss for age regression and gender classification.
    
    Computes weighted combination of:
        - Age loss: Mean Absolute Error (L1 loss)
        - Gender loss: Binary Cross-Entropy
    
    Total loss = age_weight * age_loss + gender_weight * gender_loss
    
    Args:
        age_weight: Scaling factor for age loss (default: 1.0)
        gender_weight: Scaling factor for gender loss (default: 1.0)
        
    Returns:
        Tuple of (total_loss, loss_dict) where loss_dict contains:
            - 'total': Combined weighted loss
            - 'age': Age MAE loss
            - 'gender': Gender BCE loss
    """
    
    def __init__(self, age_weight: float = 1.0, gender_weight: float = 1.0):
        super().__init__()
        self.age_weight = age_weight
        self.gender_weight = gender_weight
    
    def forward(
        self,
        age_pred: torch.Tensor,
        gender_pred: torch.Tensor,
        age_target: torch.Tensor,
        gender_target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute multi-task loss.
        
        Args:
            age_pred: Predicted ages, shape (batch_size,)
            gender_pred: Predicted gender probabilities, shape (batch_size,)
            age_target: Ground truth ages, shape (batch_size,)
            gender_target: Ground truth gender labels, shape (batch_size,)
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Ensure all tensors are 1D (already squeezed in training code)
        age_pred = age_pred.squeeze()
        gender_pred = gender_pred.squeeze()
        age_target = age_target.squeeze().float()  # Convert to float32
        gender_target = gender_target.squeeze().float()  # Convert to float32
        
        # Compute individual losses
        age_loss = F.l1_loss(age_pred, age_target)  # MAE for age regression
        gender_loss = F.binary_cross_entropy(gender_pred, gender_target)  # BCE for gender
        
        # Weighted combination
        total_loss = self.age_weight * age_loss + self.gender_weight * gender_loss
        
        # Return both total and individual losses for monitoring
        loss_dict = {
            'total': total_loss.item(),
            'age': age_loss.item(),
            'gender': gender_loss.item()
        }
        
        return total_loss, loss_dict
