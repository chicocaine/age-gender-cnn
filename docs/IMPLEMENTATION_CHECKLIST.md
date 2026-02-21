# Implementation Checklist

This checklist translates the project objectives, methodology, evaluation plan, and experiment template into actionable steps. Use it to track progress from data preparation through evaluation and reporting.

---

## 1. Research & Planning
- [x] Review key papers and open-source implementations for age estimation and gender classification.
- [x] Define the learning formulation to test (age regression vs age groups; single-task vs multi-task).
- [x] Finalize dataset usage plan (UTKFace primary, Adience for generalization).
- [x] Confirm target input size (e.g., 224×224) and normalization strategy.

---

## 2. Data Preparation
- [x] Download and verify UTKFace dataset structure and labels.
- [x] Implement or validate data loaders for UTKFace.
- [ ] Implement face detection/cropping for datasets that require it (or document that UTKFace is pre-cropped).
- [x] Implement image resizing to the target resolution.
- [x] Implement normalization (e.g., ImageNet mean/std).
- [x] Implement data augmentation:
  - [x] Horizontal flip
  - [x] Random brightness/contrast
  - [x] Mild rotation (±10°)
  - [x] Light random crop/zoom
- [x] Validate preprocessing outputs with a small sample visualization.

---

## 3. Dataset Splits & Reproducibility
- [ ] Create stratified train/validation/test splits (70/15/15) by gender and broad age bands.
- [ ] Fix and record random seed for reproducibility.
- [ ] Save split metadata (file lists and labels).

---

## 4. Model Architecture
- [ ] Implement MobileNetV2 backbone with pretrained weights.
- [ ] Implement multi-task heads:
  - [ ] Age regression head (single neuron)
  - [ ] Gender classification head (sigmoid)
- [ ] Ensure shared feature extraction and task-specific heads are correctly wired.

---

## 5. Training Setup
- [ ] Implement loss functions:
  - [ ] MAE for age regression
  - [ ] Binary Cross-Entropy for gender classification
- [ ] Implement combined loss with configurable weights.
- [ ] Configure optimizer (Adam) and learning rate (e.g., 1e-4).
- [ ] Configure batch size and epochs.
- [ ] Add training/validation loops with logging.
- [ ] Save model checkpoints and best validation model.

---

## 6. Evaluation
- [ ] Compute UTKFace test metrics:
  - [ ] Age MAE
  - [ ] Gender accuracy
  - [ ] Gender confusion matrix
- [ ] Perform cross-dataset evaluation on Adience:
  - [ ] Map age predictions to Adience age groups
  - [ ] Report age-group accuracy
- [ ] Conduct error analysis:
  - [ ] Examine incorrect predictions
  - [ ] Analyze trends by age range
  - [ ] Identify demographic/visual patterns

---

## 7. Ethics & Limitations
- [ ] Document dataset imbalance and potential bias.
- [ ] Document ambiguity in age/gender labeling.
- [ ] Document possible misuse risks and responsible use guidance.

---

## 8. Experiment Tracking
- [ ] Create a JSON configuration for each experiment.
- [ ] Fill an experiment record using the template:
  - [ ] Objective and hypothesis
  - [ ] Dataset and preprocessing details
  - [ ] Model architecture and training setup
  - [ ] Metrics and results
  - [ ] Discussion, limitations, and next steps
  - [ ] Reproducibility notes (seed, hardware, versions)
- [ ] Link related notebooks used for analysis.

---

## 9. Demonstration & Inference
- [ ] Implement inference pipeline for single-image prediction.
- [ ] Validate predictions on unseen samples.
- [ ] Provide a simple demo UI or CLI output for age and gender predictions.

---

## 10. Final Review
- [ ] Ensure all documentation is consistent across objectives, methodology, and evaluation.
- [ ] Verify that all results are reproducible and fully recorded.
- [ ] Summarize findings and align next steps with objectives.
