"""Software Defect Prediction Package."""

__version__ = "1.0.0"
__author__ = "Security Research Team"
__email__ = "research@example.com"

from .data.synthetic_data import generate_synthetic_dataset, create_train_test_split
from .models.defect_predictor import DefectPredictor
from .evaluation.metrics import DefectPredictionEvaluator
from .explainability.shap_explainer import DefectPredictionExplainer
from .features.feature_engineering import FeatureEngineer

__all__ = [
    "generate_synthetic_dataset",
    "create_train_test_split", 
    "DefectPredictor",
    "DefectPredictionEvaluator",
    "DefectPredictionExplainer",
    "FeatureEngineer"
]
