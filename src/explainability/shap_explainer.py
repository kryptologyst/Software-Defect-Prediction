"""Explainability and interpretability for software defect prediction."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

import shap
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.tree import export_text
import joblib


class DefectPredictionExplainer:
    """Comprehensive explainability for defect prediction models."""
    
    def __init__(self, model, feature_names: List[str], random_state: int = 42):
        """Initialize explainer.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            random_state: Random seed
        """
        self.model = model
        self.feature_names = feature_names
        self.random_state = random_state
        self.shap_explainer = None
        self.shap_values = None
        
    def explain_with_shap(self, X: pd.DataFrame, sample_size: int = 100) -> Dict[str, Any]:
        """Generate SHAP explanations.
        
        Args:
            X: Feature matrix
            sample_size: Number of samples for SHAP computation
            
        Returns:
            Dictionary with SHAP explanations
        """
        logger.info("Generating SHAP explanations")
        
        # Sample data for efficiency
        if len(X) > sample_size:
            sample_indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X.iloc[sample_indices]
        else:
            X_sample = X
            sample_indices = np.arange(len(X))
        
        # Create SHAP explainer based on model type
        if hasattr(self.model, 'predict_proba'):
            try:
                self.shap_explainer = shap.TreeExplainer(self.model)
                self.shap_values = self.shap_explainer.shap_values(X_sample)
            except:
                # Fallback to KernelExplainer for non-tree models
                self.shap_explainer = shap.KernelExplainer(
                    self.model.predict_proba, 
                    X_sample.sample(min(50, len(X_sample)), random_state=self.random_state)
                )
                self.shap_values = self.shap_explainer.shap_values(X_sample)
        else:
            # For models without predict_proba
            self.shap_explainer = shap.KernelExplainer(
                self.model.predict, 
                X_sample.sample(min(50, len(X_sample)), random_state=self.random_state)
            )
            self.shap_values = self.shap_explainer.shap_values(X_sample)
        
        # Handle multi-class case
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]  # Use positive class
        
        return {
            'shap_values': self.shap_values,
            'explainer': self.shap_explainer,
            'sample_indices': sample_indices,
            'feature_names': self.feature_names
        }
    
    def plot_shap_summary(self, X: pd.DataFrame, max_display: int = 15, 
                          save_path: Optional[str] = None):
        """Plot SHAP summary plot.
        
        Args:
            X: Feature matrix
            max_display: Maximum number of features to display
            save_path: Path to save the plot
        """
        if self.shap_explainer is None:
            self.explain_with_shap(X)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, X, feature_names=self.feature_names, 
                         max_display=max_display, show=False)
        plt.title('SHAP Summary Plot')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_shap_waterfall(self, X: pd.DataFrame, instance_idx: int = 0, 
                           save_path: Optional[str] = None):
        """Plot SHAP waterfall plot for a specific instance.
        
        Args:
            X: Feature matrix
            instance_idx: Index of instance to explain
            save_path: Path to save the plot
        """
        if self.shap_explainer is None:
            self.explain_with_shap(X)
        
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(self.shap_explainer.expected_value, 
                           self.shap_values[instance_idx], 
                           X.iloc[instance_idx], 
                           show=False)
        plt.title(f'SHAP Waterfall Plot - Instance {instance_idx}')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_shap_dependence(self, X: pd.DataFrame, feature_idx: int = 0, 
                            save_path: Optional[str] = None):
        """Plot SHAP dependence plot for a specific feature.
        
        Args:
            X: Feature matrix
            feature_idx: Index of feature to plot
            save_path: Path to save the plot
        """
        if self.shap_explainer is None:
            self.explain_with_shap(X)
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feature_idx, self.shap_values, X, 
                            feature_names=self.feature_names, show=False)
        plt.title(f'SHAP Dependence Plot - {self.feature_names[feature_idx]}')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_feature_importance_shap(self) -> pd.Series:
        """Get feature importance based on SHAP values.
        
        Returns:
            Feature importance scores
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call explain_with_shap first.")
        
        # Calculate mean absolute SHAP values
        importance_scores = np.mean(np.abs(self.shap_values), axis=0)
        
        return pd.Series(
            importance_scores, 
            index=self.feature_names
        ).sort_values(ascending=False)
    
    def explain_instance(self, X: pd.DataFrame, instance_idx: int = 0) -> Dict[str, Any]:
        """Explain a specific instance prediction.
        
        Args:
            X: Feature matrix
            instance_idx: Index of instance to explain
            
        Returns:
            Dictionary with instance explanation
        """
        if self.shap_explainer is None:
            self.explain_with_shap(X)
        
        instance = X.iloc[instance_idx]
        shap_value = self.shap_values[instance_idx]
        
        # Get prediction
        if hasattr(self.model, 'predict_proba'):
            prediction = self.model.predict_proba(instance.values.reshape(1, -1))[0]
        else:
            prediction = self.model.predict(instance.values.reshape(1, -1))[0]
        
        # Create explanation
        explanation = {
            'instance': instance.to_dict(),
            'shap_values': dict(zip(self.feature_names, shap_value)),
            'prediction': prediction,
            'expected_value': self.shap_explainer.expected_value
        }
        
        # Sort features by absolute SHAP value
        feature_importance = sorted(
            explanation['shap_values'].items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        explanation['feature_importance'] = feature_importance
        
        return explanation
    
    def generate_explanation_report(self, X: pd.DataFrame, 
                                  top_features: int = 10) -> str:
        """Generate a comprehensive explanation report.
        
        Args:
            X: Feature matrix
            top_features: Number of top features to include
            
        Returns:
            Formatted explanation report
        """
        if self.shap_explainer is None:
            self.explain_with_shap(X)
        
        # Get feature importance
        feature_importance = self.get_feature_importance_shap()
        
        # Get model predictions
        if hasattr(self.model, 'predict_proba'):
            predictions = self.model.predict_proba(X)[:, 1]
        else:
            predictions = self.model.predict(X)
        
        # Calculate statistics
        mean_prediction = np.mean(predictions)
        high_risk_count = np.sum(predictions > 0.7)
        
        report = f"""
Defect Prediction Model Explanation Report
{'=' * 50}

Model Overview:
- Total instances analyzed: {len(X)}
- Mean defect probability: {mean_prediction:.3f}
- High-risk instances (>70%): {high_risk_count}

Top {top_features} Most Important Features:
"""
        
        for i, (feature, importance) in enumerate(feature_importance.head(top_features).items()):
            report += f"{i+1:2d}. {feature:<25} {importance:.4f}\n"
        
        report += f"""
SHAP Analysis:
- Expected value (baseline): {self.shap_explainer.expected_value:.4f}
- Feature interactions captured: Yes
- Explanation method: TreeExplainer/KernelExplainer

Key Insights:
- Features with positive SHAP values increase defect probability
- Features with negative SHAP values decrease defect probability
- Feature importance is based on mean absolute SHAP values
"""
        
        return report
    
    def plot_feature_interactions(self, X: pd.DataFrame, 
                                feature_pairs: List[Tuple[str, str]] = None,
                                save_path: Optional[str] = None):
        """Plot feature interaction plots.
        
        Args:
            X: Feature matrix
            feature_pairs: List of feature pairs to plot
            save_path: Path to save the plot
        """
        if self.shap_explainer is None:
            self.explain_with_shap(X)
        
        if feature_pairs is None:
            # Use top 4 features for interactions
            top_features = self.get_feature_importance_shap().head(4)
            feature_pairs = [(f1, f2) for i, f1 in enumerate(top_features.index) 
                           for f2 in top_features.index[i+1:]]
        
        n_pairs = len(feature_pairs)
        if n_pairs == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, (feat1, feat2) in enumerate(feature_pairs[:4]):
            if i >= 4:
                break
                
            try:
                feat1_idx = self.feature_names.index(feat1)
                feat2_idx = self.feature_names.index(feat2)
                
                shap.dependence_plot(
                    feat1_idx, self.shap_values, X,
                    interaction_index=feat2_idx,
                    feature_names=self.feature_names,
                    ax=axes[i],
                    show=False
                )
                axes[i].set_title(f'{feat1} vs {feat2}')
            except (ValueError, IndexError):
                axes[i].text(0.5, 0.5, f'Feature not found:\n{feat1} or {feat2}', 
                           ha='center', va='center', transform=axes[i].transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_explanations(self, X: pd.DataFrame, filepath: str):
        """Save explanations to file.
        
        Args:
            X: Feature matrix
            filepath: Path to save explanations
        """
        if self.shap_explainer is None:
            self.explain_with_shap(X)
        
        explanations = {
            'shap_values': self.shap_values,
            'expected_value': self.shap_explainer.expected_value,
            'feature_names': self.feature_names,
            'model_type': type(self.model).__name__
        }
        
        joblib.dump(explanations, filepath)
        logger.info(f"Explanations saved to {filepath}")
    
    @classmethod
    def load_explanations(cls, filepath: str):
        """Load explanations from file.
        
        Args:
            filepath: Path to load explanations from
            
        Returns:
            Loaded explanations dictionary
        """
        explanations = joblib.load(filepath)
        logger.info(f"Explanations loaded from {filepath}")
        return explanations


def explain_defect_prediction(model, X: pd.DataFrame, y: np.ndarray = None,
                            feature_names: List[str] = None) -> DefectPredictionExplainer:
    """Convenience function to create explainer for defect prediction.
    
    Args:
        model: Trained model
        X: Feature matrix
        y: Target labels (optional)
        feature_names: List of feature names
        
    Returns:
        DefectPredictionExplainer instance
    """
    if feature_names is None:
        feature_names = X.columns.tolist()
    
    explainer = DefectPredictionExplainer(model, feature_names)
    
    # Generate explanations
    explainer.explain_with_shap(X)
    
    return explainer


if __name__ == "__main__":
    # Test explainer
    from ..data.synthetic_data import generate_synthetic_dataset
    from ..models.defect_predictor import DefectPredictor
    
    # Generate sample data
    data = generate_synthetic_dataset(n_samples=200)
    
    # Train a model
    predictor = DefectPredictor(model_type="random_forest")
    predictor.fit(data['X'], data['y'])
    
    # Create explainer
    explainer = DefectPredictionExplainer(predictor.model, data['feature_names'])
    
    # Generate explanations
    explanations = explainer.explain_with_shap(data['X'])
    
    # Print report
    report = explainer.generate_explanation_report(data['X'])
    print(report)
    
    # Show feature importance
    importance = explainer.get_feature_importance_shap()
    print("\nTop 10 features by SHAP importance:")
    print(importance.head(10))
