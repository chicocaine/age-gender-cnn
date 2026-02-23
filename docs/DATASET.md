# Dataset Documentation

## Overview

This project uses two facial image datasets for training and evaluating the age-gender CNN:

1. **UTKFace** (Primary): Training and evaluation dataset
2. **Adience** (Secondary): Cross-dataset generalization testing

---

## UTKFace Dataset

### Description

The UTKFace dataset is a large-scale face dataset with age, gender, and ethnicity annotations. It consists of over 20,000 face images with age ranges from 0 to 116 years.

**Key Features**:
- **Size**: ~23,000 images
- **Age Range**: 0-116 years (continuous values)
- **Gender**: Binary (0 = Male, 1 = Female)
- **Ethnicity**: 5 categories (White, Black, Asian, Indian, Others)
- **Source**: In-the-wild images from internet sources

### Download Instructions

1. Visit the UTKFace dataset page: [https://susanqq.github.io/UTKFace/](https://susanqq.github.io/UTKFace/)
2. Download the aligned & cropped faces dataset
3. Extract to `dataset/raw/UTKFace/`

### File Naming Convention

UTKFace images follow a structured filename format:

```
[age]_[gender]_[race]_[date&time].jpg
```

**Example**: `25_1_0_20170109212345678.jpg`
- Age: 25 years
- Gender: 1 (Female)
- Race: 0 (White)
- Timestamp: 20170109212345678

### Parsing Logic

The filename parsing regex used in the project:

```python
pattern = r'^(?P<age>\d+)_(?P<gender>[01])(?:_(?P<rest>.+))?$'
```

**Handling Invalid Filenames**:
- Files not matching the pattern are logged to `dataset/processed/utkface_invalid_filenames_*.csv`
- These files are excluded from training/evaluation

### Preprocessing

1. **Metadata Generation** (Notebook 01):
   - Parse filenames to extract age and gender
   - Generate metadata CSV with columns: `filename`, `age`, `gender`
   - Save to `dataset/processed/utkface_metadata.csv`

2. **Data Splitting**:
   - 70% training, 15% validation, 15% test
   - Stratified by gender and age bands to ensure balanced distribution
   - Splits saved to `dataset/processed/utkface_splits.json`

3. **Image Preprocessing**:
   - Images are already cropped and aligned (224×224 recommended)
   - Apply ImageNet normalization: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`
   - Data augmentation applied during training (see METHODOLOGY.md)

---

## Adience Dataset

### Description

The Adience Benchmark for Age and Gender Classification dataset contains images from Flickr albums.

**Key Features**:
- **Size**: ~26,000 images
- **Age Groups**: 8 bins `['0-2', '4-6', '8-13', '15-20', '25-32', '38-43', '48-53', '60+']`
- **Gender**: Binary (male, female)
- **Challenge**: Unfiltered faces with extreme variations in pose, lighting, and occlusion

### Download Instructions

1. Visit: [https://talhassner.github.io/home/projects/Adience/Adience-data.html](https://talhassner.github.io/home/projects/Adience/Adience-data.html)
2. Download aligned face images
3. Extract to `dataset/raw/Adience/`

### Usage in Project

**Primary Purpose**: Cross-dataset evaluation to test model generalization.

**Age Bin Mapping**:
Since Adience uses age bins while our model predicts continuous ages, we map predictions to bins:

```python
def map_age_to_adience_bin(age: float) -> str:
    if age <= 3: return '0-2'
    elif age <= 7: return '4-6'
    elif age <= 14: return '8-13'
    elif age <= 22: return '15-20'
    elif age <= 35: return '25-32'
    elif age <= 45: return '38-43'
    elif age <= 56: return '48-53'
    else: return '60+'
```

---

## Dataset Statistics

### UTKFace (After Filtering)

```
Total Samples: ~23,000
Age Distribution:
  0-20 years:   ~25%
  21-40 years:  ~50%
  41-60 years:  ~20%
  61+ years:    ~5%

Gender Distribution:
  Male (0):     ~54%
  Female (1):   ~46%
```

### Class Imbalance Considerations

1. **Age**: Significant imbalance with fewer samples for elderly
2. **Gender**: Relatively balanced
3. **Mitigation**: Stratified splitting ensures proportional representation in train/val/test

---

## Data Pipeline

```
Raw Images (.jpg files)
    ↓
Notebook 01: Dataset Exploration
    ├── Parse filenames
    ├── Generate metadata CSV
    └── Export valid/invalid file lists
    ↓
Notebook 04: Model Experiments
    ├── Load metadata
    ├── Create stratified splits (70/15/15)
    ├── Apply preprocessing transforms
    └── DataLoader creation
    ↓
Training & Evaluation
```

---

## File Structure

```
dataset/
├── raw/
│   ├── UTKFace/          # Original UTKFace images
│   │   └── *.jpg
│   └── Adience/          # Optional: Adience images
│       └── *.jpg
├── processed/
│   ├── utkface_metadata.csv                    # Parsed metadata
│   ├── utkface_invalid_filenames_*.csv         # Invalid files log
│   ├── utkface_exploration_manifest_*.json     # Exploration summary
│   └── utkface_splits.json                     # Train/val/test indices
└── README.md
```

---

## Data Quality Notes

**UTKFace Limitations**:
- Some mislabeled samples (age or gender errors)
- Varying image quality and lighting conditions
- Potential ethnic/geographic bias
- Binary gender labels only

**Recommendations**:
- Manual inspection of outlier predictions during error analysis
- Consider age uncertainty ranges rather than point estimates
- Acknowledge dataset limitations in model deployment documentation

---

## References

1. UTKFace: Zhang, Z., Song, Y., & Qi, H. (2017). "Age Progression/Regression by Conditional Adversarial Autoencoder"
2. Adience: Eidinger, E., Enbar, R., & Hassner, T. (2014). "Age and Gender Estimation of Unfiltered Faces"

---

## License and Usage

**UTKFace**: Check original dataset license for usage restrictions
**Adience**: Check original dataset license for usage restrictions

**Important**: This project is for educational and research purposes only. Ensure compliance with dataset licenses before using for commercial applications.
