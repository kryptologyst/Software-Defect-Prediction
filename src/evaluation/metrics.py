"""Evaluation metrics and model assessment for software defect prediction."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.calibration import calibration_curve


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    auc_pr: float
    precision_at_10: float
    precision_at_20: float
    recall_at_80_precision: float
    false_positive_rate: float
    false_negative_rate: float
    specificity: float
    sensitivity: float


class DefectPredictionEvaluator:
    """Comprehensive evaluator for defect prediction models."""
    
    def __init__(self, random_state: int = 42):
        """Initialize evaluator.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.metrics_history = []
        
    def calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               y_proba: np.ndarray) -> EvaluationMetrics:
        """Calculate basic classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            
        Returns:
            EvaluationMetrics object
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # AUC metrics
        auc_roc = roc_auc_score(y_true, y_proba[:, 1]) if len(y_proba.shape) > 1 else roc_auc_score(y_true, y_proba)
        auc_pr = average_precision_score(y_true, y_proba[:, 1]) if len(y_proba.shape) > 1 else average_precision_score(y_true, y_proba)
        
        # Precision@K metrics
        precision_at_10 = self._calculate_precision_at_k(y_true, y_proba, k=10)
        precision_at_20 = self._calculate_precision_at_k(y_true, y_proba, k=20)
        
        # Recall at fixed precision
        recall_at_80_precision = self._calculate_recall_at_precision(y_true, y_proba, target_precision=0.8)
        
        # Confusion matrix metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        return EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            precision_at_10=precision_at_10,
            precision_at_20=precision_at_20,
            recall_at_80_precision=recall_at_80_precision,
            false_positive_rate=false_positive_rate,
            false_negative_rate=false_negative_rate,
            specificity=specificity,
            sensitivity=sensitivity
        )
    
    def _calculate_precision_at_k(self, y_true: np.ndarray, y_proba: np.ndarray, 
                                 k: int) -> float:
        """Calculate precision at top K predictions.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            k: Number of top predictions to consider
            
        Returns:
            Precision at K
        """
        if len(y_proba.shape) > 1:
            proba_scores = y_proba[:, 1]
        else:
            proba_scores = y_proba
            
        # Get top K indices
        top_k_indices = np.argsort(proba_scores)[-k:]
        
        # Calculate precision
        if len(top_k_indices) > 0:
            return np.mean(y_true[top_k_indices])
        else:
            return 0.0
    
    def _calculate_recall_at_precision(self, y_true: np.ndarray, y_proba: np.ndarray, 
                                      target_precision: float) -> float:
        """Calculate recall at target precision.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            target_precision: Target precision level
            
        Returns:
            Recall at target precision
        """
        if len(y_proba.shape) > 1:
            proba_scores = y_proba[:, 1]
        else:
            proba_scores = y_proba
            
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, proba_scores)
        
        # Find threshold that achieves target precision
        valid_indices = precision >= target_precision
        if np.any(valid_indices):
            best_idx = np.argmax(recall[valid_indices])
            return recall[valid_indices][best_idx]
        else:
            return 0.0
    
    def evaluate_model(self, model, X: pd.DataFrame, y: np.ndarray, 
                      cv_folds: int = 5) -> Dict[str, Any]:
        """Evaluate model with cross-validation.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target labels
            cv_folds: Number of CV folds
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating model with {cv_folds}-fold cross-validation")
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='roc_auc')
        
        # Predictions on full dataset
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        
        # Calculate metrics
        metrics = self.calculate_basic_metrics(y, y_pred, y_proba)
        
        # Store metrics
        self.metrics_history.append({
            'model': type(model).__name__,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'metrics': metrics
        })
        
        return {
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_proba
        }
    
    def create_evaluation_report(self, results: Dict[str, Any], 
                               model_name: str = "Model") -> str:
        """Create a comprehensive evaluation report.
        
        Args:
            results: Evaluation results
            model_name: Name of the model
            
        Returns:
            Formatted evaluation report
        """
        metrics = results['metrics']
        
        report = f"""
{model_name} Evaluation Report
{'=' * 50}

Cross-Validation Results:
- AUC-ROC: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}

Classification Metrics:
- Accuracy: {metrics.accuracy:.4f}
- Precision: {metrics.precision:.4f}
- Recall: {metrics.recall:.4f}
- F1-Score: {metrics.f1:.4f}

AUC Metrics:
- AUC-ROC: {metrics.auc_roc:.4f}
- AUC-PR: {metrics.auc_pr:.4f}

Business Metrics:
- Precision@10: {metrics.precision_at_10:.4f}
- Precision@20: {metrics.precision_at_20:.4f}
- Recall@80% Precision: {metrics.recall_at_80_precision:.4f}

Error Rates:
- False Positive Rate: {metrics.false_positive_rate:.4f}
- False Negative Rate: {metrics.false_negative_rate:.4f}
- Specificity: {metrics.specificity:.4f}
- Sensitivity: {metrics.sensitivity:.4f}
"""
        return report
    
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray, 
                       model_name: str = "Model", save_path: Optional[str] = None):
        """Plot ROC curve.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            model_name: Name of the model
            save_path: Path to save the plot
        """
        if len(y_proba.shape) > 1:
            proba_scores = y_proba[:, 1]
        else:
            proba_scores = y_proba
            
        fpr, tpr, _ = roc_curve(y_true, proba_scores)
        auc_score = roc_auc_score(y_true, proba_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_proba: np.ndarray, 
                                   model_name: str = "Model", save_path: Optional[str] = None):
        """Plot precision-recall curve.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            model_name: Name of the model
            save_path: Path to save the plot
        """
        if len(y_proba.shape) > 1:
            proba_scores = y_proba[:, 1]
        else:
            proba_scores = y_proba
            
        precision, recall, _ = precision_recall_curve(y_true, proba_scores)
        auc_pr = average_precision_score(y_true, proba_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'{model_name} (AUC-PR = {auc_pr:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             model_name: str = "Model", save_path: Optional[str] = None):
        """Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            save_path: Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Clean', 'Defect'], 
                   yticklabels=['Clean', 'Defect'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, feature_importance: pd.Series, 
                               model_name: str = "Model", top_k: int = 15, 
                               save_path: Optional[str] = None):
        """Plot feature importance.
        
        Args:
            feature_importance: Feature importance scores
            model_name: Name of the model
            top_k: Number of top features to show
            save_path: Path to save the plot
        """
        top_features = feature_importance.head(top_k)
        
        plt.figure(figsize=(10, 8))
        top_features.plot(kind='barh')
        plt.title(f'Top {top_k} Feature Importance - {model_name}')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_model_comparison(self, results_list: List[Dict[str, Any]], 
                              model_names: List[str]) -> pd.DataFrame:
        """Create a comparison table of multiple models.
        
        Args:
            results_list: List of evaluation results
            model_names: List of model names
            
        Returns:
            DataFrame with model comparison
        """
        comparison_data = []
        
        for results, name in zip(results_list, model_names):
            metrics = results['metrics']
            comparison_data.append({
                'Model': name,
                'AUC-ROC': metrics.auc_roc,
                'AUC-PR': metrics.auc_pr,
                'Precision': metrics.precision,
                'Recall': metrics.recall,
                'F1-Score': metrics.f1,
                'Precision@10': metrics.precision_at_10,
                'Precision@20': metrics.precision_at_20,
                'CV Mean': results['cv_mean'],
                'CV Std': results['cv_std']
            })
        
        return pd.DataFrame(comparison_data).round(4)
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame, 
                            metric: str = 'AUC-ROC', save_path: Optional[str] = None):
        """Plot model comparison for a specific metric.
        
        Args:
            comparison_df: Model comparison DataFrame
            metric: Metric to plot
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        comparison_df.plot(x='Model', y=metric, kind='bar')
        plt.title(f'Model Comparison - {metric}')
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def evaluate_defect_predictor(model, X: pd.DataFrame, y: np.ndarray, 
                            model_name: str = "Model") -> Dict[str, Any]:
    """Convenience function to evaluate a defect predictor.
    
    Args:
        model: Trained model
        X: Feature matrix
        y: Target labels
        model_name: Name of the model
        
    Returns:
        Evaluation results
    """
    evaluator = DefectPredictionEvaluator()
    results = evaluator.evaluate_model(model, X, y)
    
    # Print report
    report = evaluator.create_evaluation_report(results, model_name)
    print(report)
    
    return results


if __name__ == "__main__":
    # Test evaluator
    from ..data.synthetic_data import generate_synthetic_dataset
    from ..models.defect_predictor import DefectPredictor
    
    # Generate sample data
    data = generate_synthetic_dataset(n_samples=500)
    
    # Train a model
    predictor = DefectPredictor(model_type="random_forest")
    predictor.fit(data['X'], data['y'])
    
    # Evaluate
    evaluator = DefectPredictionEvaluator()
    results = evaluator.evaluate_model(predictor.model, data['X'], data['y'])
    
    # Print results
    report = evaluator.create_evaluation_report(results, "Random Forest")
    print(report)
