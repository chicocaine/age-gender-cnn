"""
Metrics computation for age-gender prediction.

Provides functions to calculate evaluation metrics and handle
cross-dataset evaluation with age bin mapping.
"""

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, mean_absolute_error
from typing import Dict, List, Tuple

# Ordered list of Adience age bins (index position used for tolerance metric)
ADIENCE_BIN_ORDER: List[str] = [
    '0-2', '4-6', '8-13', '15-20', '25-32', '38-43', '48-53', '60+'
]

# Strict numeric boundaries for each Adience bin
ADIENCE_BIN_RANGES: Dict[str, Tuple[float, float]] = {
    '0-2':   (0,  2),
    '4-6':   (4,  6),
    '8-13':  (8,  13),
    '15-20': (15, 20),
    '25-32': (25, 32),
    '38-43': (38, 43),
    '48-53': (48, 53),
    '60+':   (60, float('inf')),
}


def calculate_age_mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculate Mean Absolute Error for age predictions.
    
    Args:
        predictions: Predicted ages
        targets: Ground truth ages
        
    Returns:
        MAE in years
    """
    return float(mean_absolute_error(targets, predictions))


def calculate_gender_accuracy(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5
) -> float:
    """Calculate binary classification accuracy for gender.
    
    Args:
        predictions: Gender probability predictions [0, 1]
        targets: Ground truth gender labels (0 or 1)
        threshold: Probability threshold for classification
        
    Returns:
        Accuracy score [0, 1]
    """
    pred_labels = (predictions > threshold).astype(int)
    return float(accuracy_score(targets, pred_labels))


def calculate_gender_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """Calculate confusion matrix for gender classification.
    
    Args:
        predictions: Gender probability predictions [0, 1]
        targets: Ground truth gender labels (0 or 1)
        threshold: Probability threshold for classification
        
    Returns:
        2x2 confusion matrix [[TN, FP], [FN, TP]]
    """
    pred_labels = (predictions > threshold).astype(int)
    return confusion_matrix(targets, pred_labels)


def map_age_to_adience_bin(age: float) -> str:
    """Map continuous age to Adience age bin.
    
    Adience bins: ['0-2', '4-6', '8-13', '15-20', '25-32', '38-43', '48-53', '60+']
    
    Mapping strategy:
        - 0-3: '0-2'
        - 4-7: '4-6'
        - 8-14: '8-13'
        - 15-22: '15-20'
        - 23-35: '25-32'
        - 36-45: '38-43'
        - 46-56: '48-53'
        - 57+: '60+'
    
    Args:
        age: Continuous age value
        
    Returns:
        Adience age bin string
    """
    if age <= 3:
        return '0-2'
    elif age <= 7:
        return '4-6'
    elif age <= 14:
        return '8-13'
    elif age <= 22:
        return '15-20'
    elif age <= 35:
        return '25-32'
    elif age <= 45:
        return '38-43'
    elif age <= 56:
        return '48-53'
    else:
        return '60+'


def map_ages_to_bins(ages: np.ndarray) -> List[str]:
    """Map array of ages to Adience bins.
    
    Args:
        ages: Array of continuous age values
        
    Returns:
        List of bin labels
    """
    return [map_age_to_adience_bin(age) for age in ages]


def calculate_age_bin_accuracy(
    predicted_bins: List[str],
    target_bins: List[str]
) -> float:
    """Calculate accuracy for age bin classification.
    
    Args:
        predicted_bins: List of predicted age bins
        target_bins: List of ground truth age bins
        
    Returns:
        Accuracy score [0, 1]
    """
    predicted_bins = np.array(predicted_bins)
    target_bins = np.array(target_bins)
    return float(np.mean(predicted_bins == target_bins))


def calculate_metrics_by_age_range(
    age_predictions: np.ndarray,
    age_targets: np.ndarray,
    age_ranges: List[Tuple[int, int]] = [(0, 20), (21, 40), (41, 60), (61, 120)]
) -> Dict[str, Dict[str, float]]:
    """Calculate age MAE for different age ranges.
    
    Args:
        age_predictions: Predicted ages
        age_targets: Ground truth ages
        age_ranges: List of (min_age, max_age) tuples
        
    Returns:
        Dictionary mapping range strings to metrics
    """
    results = {}
    
    for min_age, max_age in age_ranges:
        # Filter samples in this age range
        mask = (age_targets >= min_age) & (age_targets <= max_age)
        
        if mask.sum() == 0:
            continue
        
        range_preds = age_predictions[mask]
        range_targets = age_targets[mask]
        
        mae = calculate_age_mae(range_preds, range_targets)
        
        range_key = f"{min_age}-{max_age}"
        results[range_key] = {
            'mae': mae,
            'count': int(mask.sum())
        }
    
    return results


def calculate_metrics_by_gender(
    age_predictions: np.ndarray,
    age_targets: np.ndarray,
    gender_targets: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """Calculate age MAE separately for each gender.
    
    Args:
        age_predictions: Predicted ages
        age_targets: Ground truth ages
        gender_targets: Ground truth gender labels (0 or 1)
        
    Returns:
        Dictionary with gender-specific metrics
    """
    results = {}
    
    for gender in [0, 1]:
        mask = (gender_targets == gender)
        
        if mask.sum() == 0:
            continue
        
        gender_age_preds = age_predictions[mask]
        gender_age_targets = age_targets[mask]
        
        mae = calculate_age_mae(gender_age_preds, gender_age_targets)
        
        gender_label = 'Male' if gender == 0 else 'Female'
        results[gender_label] = {
            'mae': mae,
            'count': int(mask.sum())
        }
    
    return results


def calculate_within_range_accuracy(
    pred_ages: np.ndarray,
    true_bin_labels: List[str]
) -> float:
    """Check whether each predicted continuous age falls inside the strict
    numeric range of the corresponding ground-truth Adience bin.

    This is the primary Adience accuracy metric because it avoids penalising
    predictions for label-boundary artefacts (e.g. predicting 24 when the
    true bin is '25-32' is counted correct because 24 is adjacent; the strict
    range check rewards predictions that land inside the actual bin window).

    Args:
        pred_ages: Continuous age predictions from the model (N,).
        true_bin_labels: Ground-truth Adience bin strings, length N.

    Returns:
        Fraction of samples where pred_age is within the bin's [low, high].
    """
    correct = 0
    for pred, label in zip(pred_ages, true_bin_labels):
        low, high = ADIENCE_BIN_RANGES[label]
        if low <= pred <= high:
            correct += 1
    return correct / len(true_bin_labels) if true_bin_labels else 0.0


def calculate_bin_tolerance_accuracy(
    predicted_bins: List[str],
    target_bins: List[str],
    tolerance: int = 1
) -> float:
    """Calculate age-bin accuracy allowing ±tolerance adjacent bins.

    Adience bins have gaps (e.g. ages 3, 7, 21-24 fall between bins), so a
    predicted bin that is one position away from the true bin is semantically
    much closer than a prediction several bins away. This metric counts a
    prediction as correct if its index in ADIENCE_BIN_ORDER differs from the
    true bin's index by at most `tolerance`.

    Args:
        predicted_bins: Predicted Adience bin strings, length N.
        target_bins: Ground-truth Adience bin strings, length N.
        tolerance: Maximum allowed index distance (default 1 = ±1 bin).

    Returns:
        Fraction of samples within tolerance.
    """
    bin_index = {b: i for i, b in enumerate(ADIENCE_BIN_ORDER)}
    correct = sum(
        abs(bin_index.get(p, -99) - bin_index.get(t, -99)) <= tolerance
        for p, t in zip(predicted_bins, target_bins)
    )
    return correct / len(target_bins) if target_bins else 0.0
