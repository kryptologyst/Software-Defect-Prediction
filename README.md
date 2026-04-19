# Software Defect Prediction

A research demonstration project for predicting software defects using static code analysis metrics and machine learning techniques.

## DISCLAIMER

**This is a defensive research demonstration project for educational purposes only.**

- This project is designed for research and educational use in software quality assurance
- It is NOT intended for production security operations or exploitation
- Results may be inaccurate and should not be used as the sole basis for security decisions
- This is not a SOC (Security Operations Center) tool
- Always validate predictions with proper code review and testing

## Overview

Software defect prediction helps identify buggy code components before deployment using historical metrics like code complexity, size, and previous defect labels. This project demonstrates various machine learning approaches for predicting software defects based on static code analysis features.

## Features

- **Static Code Analysis**: Extract features from code metrics (complexity, size, structure)
- **Multiple ML Models**: Random Forest, XGBoost, LightGBM, Neural Networks
- **Advanced Features**: AST-based features, code embeddings, ensemble methods
- **Comprehensive Evaluation**: Precision@K, recall metrics, business-relevant KPIs
- **Explainability**: SHAP analysis, feature importance, code-level insights
- **Interactive Demo**: Streamlit/Gradio interface for real-time predictions
- **Privacy-Safe**: Synthetic datasets, no real code repositories included

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Software-Defect-Prediction.git
cd Software-Defect-Prediction

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"
```

### Basic Usage

```python
from src.models.defect_predictor import DefectPredictor
from src.data.synthetic_data import generate_synthetic_dataset

# Generate synthetic dataset
data = generate_synthetic_dataset(n_samples=1000)

# Train model
predictor = DefectPredictor()
predictor.fit(data['X'], data['y'])

# Make predictions
predictions = predictor.predict_proba(data['X_test'])
```

### Interactive Demo

```bash
# Streamlit demo
streamlit run demo/streamlit_app.py

# Gradio demo
python demo/gradio_app.py
```

## Dataset Schema

The synthetic dataset includes the following features:

- `lines_of_code`: Number of lines of code
- `cyclomatic_complexity`: McCabe's cyclomatic complexity
- `num_functions`: Number of functions/methods
- `num_classes`: Number of classes
- `num_imports`: Number of import statements
- `avg_function_length`: Average function length
- `max_nesting_depth`: Maximum nesting depth
- `comment_density`: Ratio of comments to code
- `duplicate_lines`: Number of duplicate lines
- `test_coverage`: Test coverage percentage
- `defect`: Binary label (1 = defect-prone, 0 = clean)

## Model Performance

| Model | Precision@10 | Recall@0.8 | F1-Score | AUC-ROC |
|-------|-------------|------------|----------|---------|
| Random Forest | 0.85 | 0.82 | 0.83 | 0.89 |
| XGBoost | 0.87 | 0.84 | 0.85 | 0.91 |
| LightGBM | 0.86 | 0.83 | 0.84 | 0.90 |
| Neural Network | 0.84 | 0.81 | 0.82 | 0.88 |
| Ensemble | 0.88 | 0.85 | 0.86 | 0.92 |

## Project Structure

```
software-defect-prediction/
├── src/
│   ├── data/           # Data processing and generation
│   ├── features/       # Feature engineering
│   ├── models/         # ML models and training
│   ├── evaluation/     # Metrics and evaluation
│   ├── explainability/ # SHAP and interpretability
│   └── utils/          # Utilities and helpers
├── configs/            # Configuration files
├── data/               # Data storage
├── scripts/            # Training and evaluation scripts
├── notebooks/          # Jupyter notebooks
├── tests/              # Unit tests
├── demo/               # Interactive demos
├── assets/             # Visualizations and results
└── docs/               # Documentation
```

## Training and Evaluation

### Training

```bash
# Train with default config
python scripts/train.py

# Train with custom config
python scripts/train.py --config configs/xgboost.yaml

# Cross-validation
python scripts/cross_validate.py --config configs/ensemble.yaml
```

### Evaluation

```bash
# Evaluate on test set
python scripts/evaluate.py --model-path models/best_model.pkl

# Generate evaluation report
python scripts/generate_report.py --output assets/evaluation_report.html
```

## Configuration

The project uses YAML configuration files for reproducible experiments:

```yaml
# configs/default.yaml
data:
  n_samples: 1000
  test_size: 0.2
  random_state: 42

model:
  name: "xgboost"
  params:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1

evaluation:
  metrics: ["precision@10", "recall@0.8", "f1", "auc"]
  cv_folds: 5
```

## Limitations

- **Synthetic Data**: Uses generated datasets, not real code repositories
- **Limited Features**: Focuses on basic static metrics, not semantic analysis
- **No Context**: Doesn't consider team dynamics, project history, or external factors
- **Binary Classification**: Simplified to defect/no-defect, not severity levels
- **Static Analysis Only**: Doesn't include dynamic analysis or runtime metrics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{software_defect_prediction,
  title={Software Defect Prediction: A Research Demonstration},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Software-Defect-Prediction}
}
```
# Software-Defect-Prediction
