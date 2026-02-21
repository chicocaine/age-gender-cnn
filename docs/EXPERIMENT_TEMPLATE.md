# Experiment Record Template
## Experiment ID

## Experiment Name:
`exp_XX_<short_description>`

## Date Conducted:
YYYY-MM-DD

## Related JSON Configuration:
`experiments/exp_XX_<short_description>.json`

## Related Notebook(s):
`notebooks/03_model_experiments.ipynb`
`notebooks/04_evaluation_analysis.ipynb`

---

### 1. Objective

Clearly state why **this experiment exists**.

Example:
>The objective of this experiment is to evaluate the performance of a multi-task CNN model that performs age regression and gender classification using the UTKFace dataset, serving as a baseline for subsequent comparisons.

---

### 2. Hypothesis / Research Question

Example:
>A multi-task learning approach will improve gender classification accuracy while maintaining reasonable age estimation error by sharing facial feature representations.

### 3. Experimental Configuration (Summary)
>This section summarizes the experiment. Full details are stored in the JSON configuration.

#### Dataset
- Dataset(s): UTKFace
- Train / Validation / Test Split: 70 / 15 / 15
- Label Type:
    - Age: Regression
    - Gender: Binary classification

#### Preprocessing Pipeline
- Face detection: Haar Cascade (OpenCV)
- Face alignment: None
- Image resizing: 224 × 224
- Normalization: ImageNet mean/std
- Data augmentation:
    - Horizontal flip
    - Random brightness adjustment

#### Model Architecture
- Backbone: MobileNetV2
- Learning formulation: Multi-task learning
- Output heads:
    - Age regression head (single neuron)
    - Gender classification head (sigmoid)

#### Training Setup
- Optimizer: Adam
- Learning rate: 0.0001
- Batch size: 32
- Epochs: 20
- Loss functions:
    - Age: Mean Absolute Error (MAE)
    - Gender: Binary Cross-Entropy
- Loss weights:
    - Age: 1.0
    - Gender: 1.0

### 4. Evaluation Metrics
- Gender classification accuracy
- Age estimation Mean Absolute Error (MAE)
- Confusion matrix for gender
- Error distribution by age range

### 5. Results
| Metric | Value |
| --- | --- |
| Gender Accuracy | 86.0% |
| Age MAE | 6.4 years|

(Exact values recorded in the experiment JSON file)

### 6. Exploratory Analysis
This experiment was analyzed using Jupyter notebooks to understand model behavior and error patterns.

#### Key Observations
- Age prediction error increases for extreme age ranges (children and elderly).
- Gender classification performs better on frontal, well-lit faces.
- Data imbalance in older age groups impacts regression accuracy.

#### Visualizations Used
- Age error vs ground-truth age plot
- Gender confusion matrix
- Sample predictions on unseen images

### 7. Discussion
Interpret the results in context.
Example:
>The baseline multi-task model achieves reasonable gender classification accuracy; however, age regression errors suggest that additional regularization or alternative loss weighting may be necessary. These findings motivate further experiments exploring adjusted loss weights and age group classification.

### 8. Limitations
Explicitly state what this experiment does not solve.
- Dataset bias toward young adults
- Limited performance on low-resolution images
- No explicit face alignment step

### 9. Conclusion & Next Steps
Summarize what was learned and how it informs future work.
Example:
>This experiment establishes a functional baseline pipeline for facial age and gender prediction. Based on observed limitations, the next experiment will explore age group classification and alternative learning formulations.

### 10. Reproducibility Notes
- Random seed fixed: Yes / No
- Hardware used: CPU / GPU
- Framework versions:
    - PyTorch: X.XX
    - OpenCV: X.XX

### 11. References
List only the papers or resources directly influencing this experiment.
Example:
1. Levi, G., & Hassner, T., Age and Gender Classification Using Convolutional Neural Networks, CVPR Workshops, 2015.
2. Rothe, R., Timofte, R., & Van Gool, L., DEX: Deep EXpectation of Apparent Age, ICCV, 2015.