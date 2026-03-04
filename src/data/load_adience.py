"""
Adience dataset loader for cross-dataset evaluation.

The Adience benchmark uses age-group labels (8 bins) rather than continuous
ages. Images run through the same inference preprocessing pipeline as UTKFace
so that results are directly comparable.

Fold file format (TSV, 12 columns):
    user_id | original_image | face_id | age | gender | x | y | dx | dy |
    tilt_ang | fiducial_yaw_angle | fiducial_score

Age field formats (mixed within the same file):
    - Tuple string: "(25, 32)", "(4, 6)", "(60, 100)"
    - Bare integer:  "13", "45"   (treated as continuous, mapped to nearest bin)

Image filename convention (flat directory, no subfolders):
    landmark_aligned_face.{face_id}.{original_image}

Usage:
    dataset = AdienceDataset(fold_dir, image_dir)
    loader  = DataLoader(dataset, batch_size=32, collate_fn=adience_collate_fn)
"""

import re
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from data.preprocessing import get_inference_transforms
from utils.metrics import map_age_to_adience_bin, ADIENCE_BIN_ORDER

logger = logging.getLogger(__name__)

# Regex to parse "(N, M)" tuple strings from fold files
_TUPLE_RE = re.compile(r'^\(\s*(\d+)\s*,\s*(\d+)\s*\)$')

# Map raw tuple strings to canonical bin labels
# Adience stores '60+' as "(60, 100)" in label files
_TUPLE_TO_BIN = {
    '(0, 2)':   '0-2',
    '(4, 6)':   '4-6',
    '(8, 13)':  '8-13',
    '(15, 20)': '15-20',
    '(25, 32)': '25-32',
    '(38, 43)': '38-43',
    '(48, 53)': '48-53',
    '(60, 100)':'60+',
}


def _parse_age_field(val: str) -> Optional[str]:
    """Parse the raw 'age' field from an Adience fold file.

    Handles two formats:
      • Tuple string  "(25, 32)" → canonical bin label '25-32'
      • Bare integer  "45"       → bin via map_age_to_adience_bin()
      • Anything else            → None  (row will be dropped)

    Args:
        val: Raw string value from the 'age' column.

    Returns:
        Canonical Adience bin label string, or None if unparseable.
    """
    val = str(val).strip()

    # Try direct lookup first (fast path for the common tuple format)
    if val in _TUPLE_TO_BIN:
        return _TUPLE_TO_BIN[val]

    # Try regex parse for tuple variants with unusual spacing
    m = _TUPLE_RE.match(val)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        # Reconstruct normalised key
        key = f'({lo}, {hi})'
        if key in _TUPLE_TO_BIN:
            return _TUPLE_TO_BIN[key]
        # Unknown tuple: map midpoint to nearest bin
        return map_age_to_adience_bin((lo + hi) / 2)

    # Try bare integer
    try:
        return map_age_to_adience_bin(float(val))
    except ValueError:
        return None  # 'None', 'nan', etc. — drop row


def parse_adience_folds(fold_dir: Path) -> pd.DataFrame:
    """Load and concatenate all Adience fold files from a directory.

    Only files matching ``fold_[0-9]_data.txt`` are loaded (frontal-only
    variants ``fold_frontal_*`` are excluded to keep the evaluation
    representative of real-world face variation).

    Args:
        fold_dir: Directory containing fold_*_data.txt files.

    Returns:
        DataFrame with columns:
            image_filename  – relative filename within the image root
            age_bin         – canonical Adience bin label (e.g. '25-32')
            gender          – 0 (male) or 1 (female)
    """
    fold_files = sorted(fold_dir.glob('fold_[0-9]_data.txt'))
    if not fold_files:
        raise FileNotFoundError(
            f"No fold_*_data.txt files found in {fold_dir}. "
            "Download them from: https://talhassner.github.io/home/projects/Adience/Adience-data.html"
        )

    dfs = []
    for fpath in fold_files:
        try:
            df = pd.read_csv(fpath, sep='\t', dtype=str, na_filter=False)
            dfs.append(df)
            logger.info(f"Loaded {fpath.name}: {len(df)} rows")
        except Exception as e:
            logger.warning(f"Could not read {fpath.name}: {e}")

    if not dfs:
        raise RuntimeError("All fold files failed to load.")

    combined = pd.concat(dfs, ignore_index=True)

    # --- Parse age field ---
    combined['age_bin'] = combined['age'].apply(_parse_age_field)
    n_before = len(combined)
    combined.dropna(subset=['age_bin'], inplace=True)
    n_dropped = n_before - len(combined)
    if n_dropped:
        logger.info(f"Dropped {n_dropped} rows with unparseable age labels.")

    # --- Parse gender field ---
    gender_map = {'m': 0, 'f': 1}
    combined['gender'] = combined['gender'].str.strip().str.lower().map(gender_map)
    combined.dropna(subset=['gender'], inplace=True)
    combined['gender'] = combined['gender'].astype(int)

    # --- Build image filename ---
    # Format: landmark_aligned_face.{face_id}.{original_image}
    combined['image_filename'] = (
        'landmark_aligned_face.'
        + combined['face_id'].str.strip()
        + '.'
        + combined['original_image'].str.strip()
    )

    result = combined[['image_filename', 'age_bin', 'gender']].reset_index(drop=True)
    logger.info(f"Adience dataset: {len(result)} valid samples from {len(fold_files)} fold(s).")
    return result


class AdienceDataset(Dataset):
    """PyTorch Dataset for Adience cross-dataset evaluation.

    Uses the same ``get_inference_transforms()`` pipeline as UTKFaceDataset
    so that preprocessing is identical for both datasets.

    Args:
        fold_dir:  Directory containing fold_*_data.txt annotation files.
        image_dir: Directory containing the flat JPEG image files.
        transform: Optional albumentations transform; defaults to
                   ``get_inference_transforms()``.
    """

    def __init__(
        self,
        fold_dir: Path,
        image_dir: Path,
        transform=None,
    ):
        self.image_dir = Path(image_dir)
        self.transform = transform or get_inference_transforms()
        self.metadata = parse_adience_folds(Path(fold_dir))

        # Pre-filter rows whose image file does not exist on disk
        mask = self.metadata['image_filename'].apply(
            lambda fn: (self.image_dir / fn).exists()
        )
        missing = (~mask).sum()
        if missing:
            logger.warning(
                f"{missing} Adience images not found on disk and will be skipped."
            )
        self.metadata = self.metadata[mask].reset_index(drop=True)

        logger.info(
            f"AdienceDataset ready: {len(self.metadata)} samples "
            f"(image_dir={self.image_dir})"
        )

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int):
        row = self.metadata.iloc[idx]
        img_path = self.image_dir / row['image_filename']

        image = Image.open(img_path).convert('RGB')
        image_np = np.array(image)

        augmented = self.transform(image=image_np)
        image_tensor = augmented['image']  # (C, H, W) float32 tensor

        age_bin: str = row['age_bin']
        gender: int = int(row['gender'])

        return image_tensor, age_bin, gender

    def get_bin_distribution(self) -> pd.Series:
        """Return sample counts per age bin."""
        return self.metadata['age_bin'].value_counts().reindex(
            ADIENCE_BIN_ORDER, fill_value=0
        )


def adience_collate_fn(batch):
    """DataLoader collate function that filters out None items.

    Required because AdienceDataset may return None if image loading fails
    (though missing files are pre-filtered in __init__; this is a safety net).
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    images   = torch.stack([item[0] for item in batch])
    age_bins = [item[1] for item in batch]
    genders  = torch.tensor([item[2] for item in batch], dtype=torch.float32)
    return images, age_bins, genders
