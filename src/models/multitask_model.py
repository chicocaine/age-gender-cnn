"""
Multi-task neural network for age regression and gender classification.

Combines MobileNetV2 backbone with task-specific prediction heads.
"""

import torch
import torch.nn as nn

from .backbone import MobileNetV2Backbone


class AgeGenderModel(nn.Module):
    """Multi-task model for simultaneous age and gender prediction.
    
    Architecture:
        - Shared backbone: MobileNetV2 (1280-dim features)
        - Age head: FC(1280→128) → ReLU → Dropout(0.2) → FC(128→1)
        - Gender head: FC(1280→64) → ReLU → Dropout(0.2) → FC(64→1) → Sigmoid
    
    Args:
        pretrained_backbone: Whether to use ImageNet pretrained weights
        freeze_backbone: Whether to freeze backbone during training
        dropout_rate: Dropout probability for task heads (default: 0.2)
        
    Returns:
        Tuple of (age_prediction, gender_prediction) where:
            - age_prediction: (batch_size, 1) continuous age values
            - gender_prediction: (batch_size, 1) gender probabilities [0, 1]
    """
    
    def __init__(
        self, 
        pretrained_backbone: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        
        # Shared feature extractor
        self.backbone = MobileNetV2Backbone(
            pretrained=pretrained_backbone,
            freeze=freeze_backbone
        )
        
        # Age regression head
        self.age_head = nn.Sequential(
            nn.Linear(self.backbone.output_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 1)
        )
        
        # Gender classification head
        self.gender_head = nn.Sequential(
            nn.Linear(self.backbone.output_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Forward pass through multi-task model.
        
        Args:
            x: Input images of shape (batch_size, 3, 224, 224)
            
        Returns:
            Tuple (age_pred, gender_pred) where:
                - age_pred: (batch_size, 1) predicted ages
                - gender_pred: (batch_size, 1) gender probabilities
        """
        # Extract shared features
        features = self.backbone(x)
        
        # Task-specific predictions
        age_pred = self.age_head(features)
        gender_pred = self.gender_head(features)
        
        return age_pred, gender_pred
    
    def freeze_backbone(self):
        """Freeze backbone for fine-tuning only task heads."""
        self.backbone.freeze()
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for end-to-end training."""
        self.backbone.unfreeze()
