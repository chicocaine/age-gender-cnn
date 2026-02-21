### Methodology
---
#### A. Overview of the Approach
This study adopts a **supervised, multi-task learning approach** to explore computer vision techniques for facial age estimation and gender classification. A complete computer vision pipeline is designed, beginning with facial image preprocessing and culminating in a deep learning model capable of predicting both age and gender from a single facial image.

This approach aims to emphasize **understanding the role of each pipeline component**, rather than achieving state-of-the-art performance.

---
#### B. Dataset Description
##### B.1 Primary Dataset: UTKFace
The UTKFace dataset is used as the primary dataset for training and evaluation. It consists of over 20,000 facial images annotated with age and gender labels. The availability of exact age values enables formulation of age estimation as a regression problem, while gender labels support binary classification.
##### B.2 Secondary Dataset: Adience (Generalization Test)
The Adience dataset is used for additional testing to evaluate model generalization under unconstrained, real-world conditions. Since Adience provides age-group labels rather than exact ages, predicted age values are mapped to corresponding age groups for comparison using fixed bins (e.g., 0–2, 4–6, 8–13, 15–20, 25–32, 38–43, 48–53, 60+).

---
#### C. Data Preprocessing
#### C.1 Face Detection and Cropping
UTKFace images are already cropped around faces however the conceptual pipeline includes a face detection stage to ensure generalizability to real-world inputs. For datasets requiring it, detected facial regions are extracted and used as model input.
##### C.2 Image Resizing and Normalization
All facial images are resized to a fixed resolution (e.g., 224×224 pixels) to ensure consistency with the CNN input requirements. Pixel values are normalized to a standard range to stabilize training and improve convergence.
##### C.3 Data Augmentation
Data augmentation is applied during training to improve robustness, including horizontal flipping, random brightness and contrast adjustments, slight rotations (e.g., ±10°), and light random crops or zooms. Augmentations are kept mild to preserve facial identity cues relevant to age and gender.

---
#### D. Model Architecture 
##### D.1 Feature Extraction Backbone
The proposed model employs a **pretrained convolutional neural network** (MobileNetV2) as a shared feature extraction backbone. Transfer learning is used to leverage learned visual representations from large-scale image datasets.
##### D.2 Multi-Task Learning Design
The model adopts a **multi-task architecture** with:
- A shared feature representation extracted by the CNN backbone
- Two task-specific output heads:
    - An **age regression head** producing a continuous age prediction
    - A **gender classification head** producing a binary prediction
This approach enables shared learning of facial features while allowing specialization for each task.

---
#### E. Training Procedure
##### E.1 Learning Formulation
- **Age estimation** is treated as a regression problem.
- **Gender classification** is treated as a binary classification problem
##### E.2 Loss Functions
Separate loss functions are used for each task:
- Mean Absolute Error (MAE) for age estimation
- Binary Cross-Entropy loss for gender classification
The total loss is computed as the sum of both task losses, enabling joint optimization during training.
##### E.3 Optimization
The model is trained using gradient-based optimization techniques over multiple epochs. Training and validation splits are used to monitor performance and prevent overfitting.