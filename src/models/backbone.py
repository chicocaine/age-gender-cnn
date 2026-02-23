"""
MobileNetV2 backbone for feature extraction.

Provides pretrained MobileNetV2 backbone with optional weight freezing
for transfer learning on age-gender prediction tasks.
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class MobileNetV2Backbone(nn.Module):
    """MobileNetV2 feature extractor.
    
    Loads pretrained MobileNetV2 from torchvision and removes the
    final classification layer, returning 1280-dimensional features.
    
    Args:
        pretrained: Whether to load ImageNet pretrained weights
        freeze: Whether to freeze backbone weights for feature extraction only
        
    Returns:
        Feature tensor of shape (batch_size, 1280)
    """
    
    def __init__(self, pretrained: bool = True, freeze: bool = False):
        super().__init__()
        
        # Load pretrained MobileNetV2
        if pretrained:
            weights = MobileNet_V2_Weights.IMAGENET1K_V1
            self.model = mobilenet_v2(weights=weights)
        else:
            self.model = mobilenet_v2(weights=None)
        
        # Remove classifier (keep only features)
        self.features = self.model.features
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Freeze weights if requested
        if freeze:
            self.freeze()
        
        self.output_dim = 1280  # MobileNetV2 feature dimension
    
    def freeze(self):
        """Freeze all backbone parameters."""
        for param in self.features.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """Unfreeze all backbone parameters."""
        for param in self.features.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        """Extract features from input images.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Features of shape (batch_size, 1280)
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
