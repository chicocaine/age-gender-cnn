# End-to-End Pipeline Documentation

## Pipeline Overview

This document describes the complete workflow from raw data to trained model evaluation.

```
┌─────────────────────────────────────────────────────────────────┐
│                     AGE-GENDER CNN PIPELINE                       │
└─────────────────────────────────────────────────────────────────┘

Phase 1: Data Preparation
   ┌──────────────┐
   │  Raw Images  │ (UTKFace dataset)
   └──────┬───────┘
          │
          ↓
   ┌────────────────────────┐
   │ Notebook 01: Explore   │
   │ - Parse filenames      │
   │ - Generate metadata    │
   │ - Quality checks       │
   └───────┬────────────────┘
           │
           ↓
   ┌──────────────────┐
   │  Metadata CSV    │
   └──────────────────┘

Phase 2: Preprocessing
   ┌──────────────────────┐
   │ Notebook 02 & 03     │
   │ - Face detection     │
   │ - Resize & normalize │
   │ - Augmentation tests │
   └──────────────────────┘

Phase 3: Model Development
   ┌─────────────────────────────────┐
   │ src/models/                     │
   │ ├── backbone.py (MobileNetV2)   │
   │ ├── multitask_model.py          │
   │ └── losses.py (MAE + BCE)       │
   └─────────────────────────────────┘

Phase 4: Training
   ┌─────────────────────────────────┐
   │ Notebook 04: Experiments        │
   │ - Create dataset splits         │
   │ - Configure hyperparameters     │
   │ - Train multiple models         │
   │ - Save checkpoints              │
   └───────┬─────────────────────────┘
           │
           ↓
   ┌──────────────────────┐
   │  Model Checkpoints   │
   │  Experiment JSONs    │
   └──────────────────────┘

Phase 5: Evaluation
   ┌─────────────────────────────────┐
   │ Notebook 05: Analysis           │
   │ - Load best model               │
   │ - Test set evaluation           │
   │ - Error analysis                │
   │ - Visualizations                │
   │ - Ethical considerations        │
   └─────────────────────────────────┘

Phase 6: Deployment (Optional)
   ┌─────────────────────────────────┐
   │ ui/app.py                       │
   │ - Gradio/Streamlit interface    │
   │ - Real-time inference           │
   └─────────────────────────────────┘
```

---

## Detailed Phase Breakdown

### Phase 1: Data Preparation

**Objective**: Transform raw images into structured training data.

**Inputs**:
- Raw UTKFace images (`dataset/raw/UTKFace/*.jpg`)

**Process**:
1. **Filename Parsing**:
   ```python
   # Extract: [age]_[gender]_[race]_[timestamp].jpg
   age, gender, race, timestamp = parse_filename(filename)
   ```

2. **Metadata Generation**:
   - Create DataFrame with columns: `filename`, `age`, `gender`
   - Filter invalid filenames
   - Export to CSV

3. **Statistics & Validation**:
   - Age distribution analysis
   - Gender balance check
   - Identify outliers (age > 100, etc.)

**Outputs**:
- `dataset/processed/utkface_metadata.csv`
- `dataset/processed/utkface_invalid_filenames_*.csv`
- `dataset/processed/utkface_exploration_manifest_*.json`

**Notebook**: `notebooks/01_dataset_exploration.ipynb`

---

### Phase 2: Preprocessing Prototyping

**Objective**: Validate image preprocessing pipeline.

**Process**:
1. **Notebook 02** (With face detection):
   - Test OpenCV Haar Cascades
   - Test MediaPipe face detection
   - Crop to face region
   - Resize to 224×224
   - Apply augmentations

2. **Notebook 03** (Without face detection):
   - Direct resize (UTKFace already cropped)
   - ImageNet normalization
   - Augmentation pipeline

**Augmentation Strategy**:
```python
Transforms:
  - HorizontalFlip(p=0.5)
  - RandomBrightnessContrast(±0.2, p=0.6)
  - Rotate(±10°, p=0.5)
  - RandomResizedCrop(scale=0.9-1.0, p=0.6)
  - Normalize(ImageNet mean/std)
  - ToTensor()
```

**Outputs**:
- Validated preprocessing functions
- Augmentation samples for review

**Notebooks**: `notebooks/02_preprocessing_tests.ipynb`, `notebooks/03_preprocessing_test.ipynb`

---

### Phase 3: Model Architecture Design

**Objective**: Implement multi-task CNN architecture.

**Components**:

1. **Backbone** (`src/models/backbone.py`):
   ```
   MobileNetV2 (pretrained ImageNet)
   ├── Input: (batch, 3, 224, 224)
   ├── Features: Convolutional layers
   ├── GlobalAvgPool
   └── Output: (batch, 1280)
   ```

2. **Multi-Task Heads** (`src/models/multitask_model.py`):
   ```
   Shared Features (1280-dim)
   ├── Age Head:
   │   ├── Linear(1280 → 128)
   │   ├── ReLU + Dropout(0.2)
   │   ├── Linear(128 → 1)
   │   └── Output: age (continuous)
   │
   └── Gender Head:
       ├── Linear(1280 → 64)
       ├── ReLU + Dropout(0.2)
       ├── Linear(64 → 1)
       ├── Sigmoid
       └── Output: gender_prob [0, 1]
   ```

3. **Loss Function** (`src/models/losses.py`):
   ```python
   MultiTaskLoss:
     total_loss = α * age_loss + β * gender_loss
     age_loss = MAE(age_pred, age_true)
     gender_loss = BCE(gender_pred, gender_true)
   ```

**Parameters**:
- Total parameters: ~2.5M
- Trainable: ~2.5M (backbone not frozen)

---

### Phase 4: Training Pipeline

**Objective**: Train models with different hyperparameter configurations.

**Process**:

1. **Dataset Splitting**:
   ```python
   stratify_by = gender + age_band  # Ensure balanced splits
   train (70%), val (15%), test (15%)
   ```

2. **DataLoader Creation**:
   ```python
   train_loader = DataLoader(
       train_dataset,
       batch_size=32,
       shuffle=True,
       num_workers=2
   )
   ```

3. **Training Loop**:
   ```python
   for epoch in range(num_epochs):
       train_losses = train_one_epoch(model, train_loader, ...)
       val_losses = validate(model, val_loader, ...)
       
       if val_loss < best_val_loss:
           save_checkpoint(model, optimizer, epoch)
   ```

4. **Experiments**:
   - **Exp 01**: Baseline (lr=1e-4, full augmentation)
   - **Exp 02**: Age-focused (age_weight=2.0)
   - **Exp 03**: Conservative (lr=5e-5)
   - **Exp 04**: Minimal augmentation

**Hyperparameters**:
```python
{
  "learning_rate": 1e-4,
  "batch_size": 32,
  "num_epochs": 20,
  "optimizer": "Adam",
  "age_loss_weight": 1.0,
  "gender_loss_weight": 1.0,
  "augmentation": "full"
}
```

**Outputs**:
- `models/exp{N}_best.pth` (model checkpoints)
- `experiments/exp{N}_results.json` (metrics & history)
- `experiments/exp{N}_config.json` (configuration)
- `experiments/exp{N}_history.png` (training curves)

**Notebook**: `notebooks/04_model_experiments.ipynb`

---

### Phase 5: Evaluation & Analysis

**Objective**: Comprehensive model assessment.

**Process**:

1. **Load Best Model**:
   ```python
   best_model = select_best_experiment(experiments)
   checkpoint = torch.load(checkpoint_path)
   model.load_state_dict(checkpoint['model_state_dict'])
   ```

2. **Test Set Evaluation**:
   ```python
   age_preds, gender_preds, metrics = evaluate_model(
       model, test_loader, device
   )
   ```

3. **Metrics Computation**:
   - Age MAE (Mean Absolute Error)
   - Gender accuracy
   - Confusion matrix
   - Per-demographic performance

4. **Error Analysis**:
   - Age error distribution (histogram)
   - Predictions vs actual (scatter plot)
   - Worst predictions (failure cases)
   - Error by age range
   - Error by gender

5. **Visualizations**:
   - Training curves
   - Confusion matrix heatmap
   - Age error plots
   - Sample prediction grid

6. **Ethical Documentation**:
   - Dataset biases
   - Model limitations
   - Responsible use guidelines

**Outputs**:
- `experiments/gender_confusion_matrix.png`
- `experiments/age_error_distribution.png`
- `experiments/age_predictions_vs_actual.png`
- `experiments/sample_predictions.png`
- `experiments/age_metrics_by_range.png`

**Notebook**: `notebooks/05_evaluation_analysis.ipynb`

---

### Phase 6: Deployment (Optional)

**Objective**: Create user-facing inference interface.

**Options**:

1. **Gradio App**:
   ```python
   import gradio as gr
   
   def predict(image):
       age, gender = model.predict(image)
       return age, gender
   
   gr.Interface(predict, inputs="image", outputs=["number", "label"]).launch()
   ```

2. **Streamlit App**:
   ```python
   import streamlit as st
   
   uploaded_file = st.file_uploader("Upload face image")
   if uploaded_file:
       age, gender = model.predict(uploaded_file)
       st.write(f"Age: {age}, Gender: {gender}")
   ```

3. **REST API** (Flask/FastAPI):
   ```python
   @app.post("/predict")
   def predict(file: UploadFile):
       image = load_image(file)
       age, gender = model.predict(image)
       return {"age": age, "gender": gender}
   ```

**File**: `ui/app.py`

---

## Code Organization

```
src/
├── data/
│   ├── load_utkface.py       # PyTorch Dataset class
│   ├── load_adience.py       # Adience loader (optional)
│   └── preprocessing.py      # Transform pipelines
├── models/
│   ├── backbone.py           # MobileNetV2 feature extractor
│   ├── multitask_model.py    # Age-gender multi-task model
│   └── losses.py             # Multi-task loss function
├── training/
│   ├── train.py              # Training loop
│   ├── validate.py           # Validation loop
│   └── evaluate.py           # Evaluation metrics
├── utils/
│   ├── config.py             # Experiment configuration
│   ├── metrics.py            # Metric calculation functions
│   └── visualization.py      # Plotting utilities
└── inference/
    └── predict.py            # Inference pipeline
```

---

## Running the Pipeline

### Prerequisites

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download UTKFace dataset to `dataset/raw/UTKFace/`

### Step-by-Step Execution

1. **Explore Dataset**:
   ```bash
   jupyter notebook notebooks/01_dataset_exploration.ipynb
   ```

2. **Validate Preprocessing** (optional):
   ```bash
   jupyter notebook notebooks/02_preprocessing_tests.ipynb
   jupyter notebook notebooks/03_preprocessing_test.ipynb
   ```

3. **Train Models**:
   ```bash
   jupyter notebook notebooks/04_model_experiments.ipynb
   ```
   - Runs ~3-5 hours on GPU for 4 experiments × 20 epochs

4. **Evaluate Best Model**:
   ```bash
   jupyter notebook notebooks/05_evaluation_analysis.ipynb
   ```

5. **Deploy (optional)**:
   ```bash
   python ui/app.py
   ```

### Expected Timeline

- Data preparation: 30 minutes
- Training (4 experiments): 3-5 hours (GPU) or 15-20 hours (CPU)
- Evaluation: 15 minutes
- Total: ~4-6 hours (with GPU)

---

## Key Design Decisions

1. **Transfer Learning**: Use pretrained MobileNetV2 for faster convergence
2. **Multi-Task Learning**: Share features between age/gender tasks
3. **Stratified Splitting**: Ensure demographic balance across splits
4. **Loss Weighting**: Equal weights baseline, tunable per experiment
5. **Augmentation**: Mild transforms to preserve facial identity
6. **Checkpointing**: Save best model by validation loss
7. **Reproducibility**: Fixed random seed (42) throughout

---

## Performance Expectations

**Typical Results** (UTKFace test set):
- Age MAE: 5-8 years
- Gender Accuracy: 85-92%

**Notes**:
- Performance varies by age range (worse at extremes)
- Cross-dataset evaluation (Adience) typically shows degradation
- Real-world performance depends on image quality

---

## Troubleshooting

**Common Issues**:

1. **CUDA Out of Memory**:
   - Reduce batch size: `config.batch_size = 16`
   - Use gradient accumulation

2. **Slow Training**:
   - Ensure GPU is being used: `torch.cuda.is_available()`
   - Reduce `num_workers` if CPU bottleneck

3. **Poor Convergence**:
   - Check learning rate (try 1e-5 to 1e-3)
   - Verify data normalization
   - Inspect training curves for overfitting

4. **Metadata File Not Found**:
   - Run notebook 01 first to generate metadata
   - Check file path in config

---

## References

- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Albumentations Docs](https://albumentations.ai/)

---

## Version History

- v1.0 (2026-02-22): Initial pipeline implementation
