#!/usr/bin/env python3
"""
Software Defect Prediction - Modernized Implementation

This is a research demonstration for educational purposes only.
Results may be inaccurate and should not be used for production security decisions.

This script demonstrates the modernized software defect prediction system
with advanced ML models, comprehensive evaluation, and explainability.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
from loguru import logger

from data.synthetic_data import generate_synthetic_dataset, create_train_test_split
from models.defect_predictor import DefectPredictor
from evaluation.metrics import DefectPredictionEvaluator
from explainability.shap_explainer import DefectPredictionExplainer


def main():
    """Main demonstration function."""
    
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
    
    logger.info("Software Defect Prediction Demo - Research Use Only")
    logger.info("=" * 60)
    
    # Generate synthetic dataset
    logger.info("Generating synthetic dataset...")
    data = generate_synthetic_dataset(
        n_samples=1000,
        defect_ratio=0.3,
        random_state=42
    )
    
    logger.info(f"Dataset generated: {len(data['features'])} samples, {data['y'].mean():.1%} defect rate")
    
    # Create train-test split
    split_data = create_train_test_split(data, test_size=0.2, random_state=42)
    
    # Train multiple models
    model_types = ["random_forest", "xgboost", "lightgbm", "ensemble"]
    results = {}
    
    evaluator = DefectPredictionEvaluator(random_state=42)
    
    for model_type in model_types:
        logger.info(f"Training {model_type} model...")
        
        # Create and train predictor
        predictor = DefectPredictor(model_type=model_type, random_state=42)
        predictor.fit(
            split_data['X_train'],
            split_data['y_train'],
            use_feature_engineering=True
        )
        
        # Evaluate model
        result = evaluator.evaluate_model(
            predictor.model,
            split_data['X_test'],
            split_data['y_test']
        )
        
        results[model_type] = {
            'predictor': predictor,
            'result': result
        }
        
        # Print key metrics
        metrics = result['metrics']
        logger.info(f"{model_type}: AUC-ROC={metrics.auc_roc:.3f}, F1={metrics.f1:.3f}, Precision@10={metrics.precision_at_10:.3f}")
    
    # Find best model
    best_model_type = max(results.keys(), key=lambda k: results[k]['result']['metrics'].auc_roc)
    best_predictor = results[best_model_type]['predictor']
    
    logger.info(f"Best model: {best_model_type} (AUC-ROC: {results[best_model_type]['result']['metrics'].auc_roc:.3f})")
    
    # Generate SHAP explanations
    logger.info("Generating SHAP explanations...")
    explainer = DefectPredictionExplainer(
        best_predictor.model,
        split_data['feature_names'],
        random_state=42
    )
    
    shap_results = explainer.explain_with_shap(split_data['X_test'], sample_size=100)
    shap_importance = explainer.get_feature_importance_shap()
    
    logger.info("Top 5 most important features:")
    for i, (feature, importance) in enumerate(shap_importance.head(5).items()):
        logger.info(f"  {i+1}. {feature}: {importance:.4f}")
    
    # Make predictions on sample new modules
    logger.info("Making predictions on sample code modules...")
    
    new_modules = pd.DataFrame({
        'lines_of_code': [150, 500, 80, 800],
        'cyclomatic_complexity': [8, 25, 3, 30],
        'num_functions': [5, 15, 2, 20],
        'num_classes': [2, 8, 1, 12],
        'num_imports': [5, 20, 2, 25],
        'avg_function_length': [30, 33, 40, 40],
        'max_nesting_depth': [3, 6, 2, 7],
        'comment_density': [0.15, 0.05, 0.25, 0.03],
        'duplicate_lines': [2, 15, 0, 25],
        'test_coverage': [0.8, 0.3, 0.9, 0.2]
    })
    
    predictions = best_predictor.predict(new_modules)
    probabilities = best_predictor.predict_proba(new_modules)
    
    logger.info("Sample predictions:")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities[:, 1])):
        risk_level = "High" if prob > 0.7 else "Medium" if prob > 0.3 else "Low"
        logger.info(f"  Module {i+1}: {prob:.1%} defect probability ({risk_level} risk)")
    
    # Print comprehensive evaluation report
    logger.info("Comprehensive Evaluation Report:")
    logger.info("-" * 40)
    
    for model_type, data in results.items():
        metrics = data['result']['metrics']
        logger.info(f"{model_type.replace('_', ' ').title()}:")
        logger.info(f"  Accuracy: {metrics.accuracy:.3f}")
        logger.info(f"  Precision: {metrics.precision:.3f}")
        logger.info(f"  Recall: {metrics.recall:.3f}")
        logger.info(f"  F1-Score: {metrics.f1:.3f}")
        logger.info(f"  AUC-ROC: {metrics.auc_roc:.3f}")
        logger.info(f"  Precision@10: {metrics.precision_at_10:.3f}")
        logger.info("")
    
    logger.info("Demo completed successfully!")
    logger.info("Remember: This is for research/educational purposes only.")


if __name__ == "__main__":
    main()

