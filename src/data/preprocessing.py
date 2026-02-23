"""
Image preprocessing utilities for age-gender CNN.

Provides transformation pipelines for training and inference,
following ImageNet normalization standards for pretrained models.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Target image size for MobileNetV2
TARGET_SIZE = (224, 224)


def get_train_transforms(
    target_size: tuple = TARGET_SIZE,
    mean: list = IMAGENET_MEAN,
    std: list = IMAGENET_STD
) -> A.Compose:
    """Get augmentation pipeline for training.
    
    Applies mild augmentations that preserve facial identity:
    - Horizontal flip (50% probability)
    - Brightness and contrast adjustment (±20%, 60% probability)
    - Rotation (±10°, 50% probability)
    - Random resized crop (90-100% scale, 60% probability)
    - ImageNet normalization
    
    Args:
        target_size: Output image dimensions (height, width)
        mean: Normalization mean values for RGB channels
        std: Normalization std values for RGB channels
        
    Returns:
        Albumentations composition pipeline
    """
    return A.Compose([
        A.Resize(height=target_size[0], width=target_size[1], interpolation=cv2.INTER_LINEAR),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.6
        ),
        A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT, p=0.5),
        A.RandomResizedCrop(
            height=target_size[0],
            width=target_size[1],
            scale=(0.9, 1.0),
            ratio=(0.95, 1.05),
            p=0.6
        ),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])


def get_inference_transforms(
    target_size: tuple = TARGET_SIZE,
    mean: list = IMAGENET_MEAN,
    std: list = IMAGENET_STD
) -> A.Compose:
    """Get preprocessing pipeline for inference (no augmentation).
    
    Applies only essential preprocessing:
    - Resize to target dimensions
    - ImageNet normalization
    
    Args:
        target_size: Output image dimensions (height, width)
        mean: Normalization mean values for RGB channels
        std: Normalization std values for RGB channels
        
    Returns:
        Albumentations composition pipeline
    """
    return A.Compose([
        A.Resize(height=target_size[0], width=target_size[1], interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])


def get_minimal_augmentation_transforms(
    target_size: tuple = TARGET_SIZE,
    mean: list = IMAGENET_MEAN,
    std: list = IMAGENET_STD
) -> A.Compose:
    """Get minimal augmentation pipeline (only horizontal flip).
    
    Useful for testing overfitting with reduced augmentation.
    
    Args:
        target_size: Output image dimensions (height, width)
        mean: Normalization mean values for RGB channels
        std: Normalization std values for RGB channels
        
    Returns:
        Albumentations composition pipeline
    """
    return A.Compose([
        A.Resize(height=target_size[0], width=target_size[1], interpolation=cv2.INTER_LINEAR),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
