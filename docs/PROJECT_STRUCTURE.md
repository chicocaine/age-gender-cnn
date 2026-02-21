```
age-gender-cnn/
│
├── notebooks/                  # Jupyter notebooks (experiments)
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_preprocessing_tests.ipynb
│   ├── 03_model_experiments.ipynb
│   └── 04_evaluation_analysis.ipynb
│
├── src/                        # Core Python source code
│   ├── data/
│   │   ├── load_utkface.py
│   │   ├── load_adience.py
│   │   └── preprocessing.py
│   │
│   ├── models/
│   │   ├── backbone.py         # MobileNet / feature extractor
│   │   ├── multitask_model.py  # Age + gender heads
│   │   └── losses.py
│   │
│   ├── training/
│   │   ├── train.py
│   │   ├── validate.py
│   │   └── evaluate.py
│   │
│   ├── inference/
│   │   └── predict.py
│   │
│   └── utils/
│       ├── metrics.py
│       ├── visualization.py
│       └── config.py
│
├── dataset/                       # Dataset storage (gitignored)
│   ├── raw/
│   ├── processed/
│   └── samples/
│
├── docs/                       # Technical documentation
│   ├── methodology.md
│   ├── pipeline.md
│   ├── dataset.md
│   ├── project_structure.md
│   └── evaluation.md
│
├── ui/                         # Simple UI
│   └── app.py
│
├── .venv/                      # Python virtual environment
│
├── requirements.txt
├── README.md
└── .gitignore

```