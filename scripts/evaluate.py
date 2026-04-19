#!/usr/bin/env python3
"""Evaluation script for software defect prediction models."""

import argparse
import sys
import os
from pathlib import Path
from typing import Dict, Any, List
import yaml
import pandas as pd
import numpy as np
from loguru import logger

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.synthetic_data import generate_synthetic_dataset, create_train_test_split
from models.defect_predictor import DefectPredictor
from evaluation.metrics import DefectPredictionEvaluator


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_multiple_models(config: Dict[str, Any]) -> pd.DataFrame:
    """Evaluate multiple models and create comparison.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Model comparison DataFrame
    """
    logger.info("Evaluating multiple models")
    
    # Extract configuration
    data_config = config['data']
    features_config = config.get('features', {})
    
    # Generate dataset
    data = generate_synthetic_dataset(
        n_samples=data_config['n_samples'],
        defect_ratio=data_config['defect_ratio'],
        random_state=data_config['random_state']
    )
    
    # Create train-test split
    split_data = create_train_test_split(
        data,
        test_size=data_config['test_size'],
        random_state=data_config['random_state']
    )
    
    # Define models to evaluate
    model_types = ["random_forest", "xgboost", "lightgbm", "neural_network", "ensemble"]
    
    results_list = []
    model_names = []
    
    evaluator = DefectPredictionEvaluator(random_state=data_config['random_state'])
    
    for model_type in model_types:
        logger.info(f"Evaluating {model_type} model")
        
        try:
            # Initialize and train model
            predictor = DefectPredictor(
                model_type=model_type,
                random_state=data_config['random_state']
            )
            
            predictor.fit(
                split_data['X_train'],
                split_data['y_train'],
                use_feature_engineering=features_config.get('feature_selection', True)
            )
            
            # Evaluate model
            results = evaluator.evaluate_model(
                predictor.model,
                split_data['X_test'],
                split_data['y_test'],
                cv_folds=config.get('evaluation', {}).get('cv_folds', 5)
            )
            
            results_list.append(results)
            model_names.append(model_type.replace('_', ' ').title())
            
            logger.info(f"{model_type} - AUC-ROC: {results['metrics'].auc_roc:.3f}")
            
        except Exception as e:
            logger.warning(f"Failed to evaluate {model_type}: {str(e)}")
            continue
    
    # Create comparison DataFrame
    comparison_df = evaluator.create_model_comparison(results_list, model_names)
    
    return comparison_df, results_list, model_names


def generate_evaluation_report(comparison_df: pd.DataFrame, 
                              results_list: List[Dict[str, Any]], 
                              model_names: List[str],
                              output_dir: str) -> None:
    """Generate comprehensive evaluation report.
    
    Args:
        comparison_df: Model comparison DataFrame
        results_list: List of evaluation results
        model_names: List of model names
        output_dir: Output directory
    """
    logger.info("Generating evaluation report")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save comparison table
    comparison_path = os.path.join(output_dir, 'model_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    
    # Generate detailed report
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("Software Defect Prediction - Model Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("MODEL COMPARISON\n")
        f.write("-" * 30 + "\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("DETAILED RESULTS\n")
        f.write("-" * 30 + "\n")
        
        evaluator = DefectPredictionEvaluator()
        
        for results, name in zip(results_list, model_names):
            f.write(f"\n{name} Model:\n")
            f.write("-" * 20 + "\n")
            
            report = evaluator.create_evaluation_report(results, name)
            f.write(report)
            f.write("\n")
        
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 30 + "\n")
        
        # Find best model for each metric
        best_auc = comparison_df.loc[comparison_df['AUC-ROC'].idxmax()]
        best_precision = comparison_df.loc[comparison_df['Precision'].idxmax()]
        best_recall = comparison_df.loc[comparison_df['Recall'].idxmax()]
        best_f1 = comparison_df.loc[comparison_df['F1-Score'].idxmax()]
        
        f.write(f"Best AUC-ROC: {best_auc['Model']} ({best_auc['AUC-ROC']:.3f})\n")
        f.write(f"Best Precision: {best_precision['Model']} ({best_precision['Precision']:.3f})\n")
        f.write(f"Best Recall: {best_recall['Model']} ({best_recall['Recall']:.3f})\n")
        f.write(f"Best F1-Score: {best_f1['Model']} ({best_f1['F1-Score']:.3f})\n")
        
        f.write("\nGENERAL RECOMMENDATIONS:\n")
        f.write("- For high precision requirements, use the model with best Precision@10\n")
        f.write("- For high recall requirements, use the model with best Recall\n")
        f.write("- For balanced performance, use the model with best F1-Score\n")
        f.write("- For overall performance, use the model with best AUC-ROC\n")
        f.write("- Consider ensemble methods for robust predictions\n")
    
    logger.info(f"Evaluation report saved to {report_path}")


def create_visualizations(comparison_df: pd.DataFrame, 
                         results_list: List[Dict[str, Any]], 
                         model_names: List[str],
                         output_dir: str) -> None:
    """Create evaluation visualizations.
    
    Args:
        comparison_df: Model comparison DataFrame
        results_list: List of evaluation results
        model_names: List of model names
        output_dir: Output directory
    """
    logger.info("Creating evaluation visualizations")
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Model comparison bar plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    metrics = ['AUC-ROC', 'Precision', 'Recall', 'F1-Score']
    
    for i, metric in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        
        comparison_df.plot(
            x='Model', 
            y=metric, 
            kind='bar', 
            ax=ax,
            color='skyblue',
            edgecolor='navy',
            alpha=0.7
        )
        
        ax.set_title(f'Model Comparison - {metric}')
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'model_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # ROC curves comparison
    plt.figure(figsize=(10, 8))
    
    from sklearn.metrics import roc_curve
    
    for results, name in zip(results_list, model_names):
        y_true = results['predictions']  # This should be actual labels
        y_proba = results['probabilities']
        
        if len(y_proba.shape) > 1:
            y_scores = y_proba[:, 1]
        else:
            y_scores = y_proba
            
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_score = results['metrics'].auc_roc
        
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(plots_dir, 'roc_curves_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Precision-Recall curves comparison
    plt.figure(figsize=(10, 8))
    
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    for results, name in zip(results_list, model_names):
        y_true = results['predictions']  # This should be actual labels
        y_proba = results['probabilities']
        
        if len(y_proba.shape) > 1:
            y_scores = y_proba[:, 1]
        else:
            y_scores = y_proba
            
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        avg_precision = average_precision_score(y_true, y_scores)
        
        plt.plot(recall, precision, label=f'{name} (AP = {avg_precision:.3f})', linewidth=2)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(plots_dir, 'pr_curves_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualizations saved to {plots_dir}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate software defect prediction models")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="results/evaluation",
        help="Output directory for results"
    )
    parser.add_argument(
        "--models", 
        nargs='+',
        default=["random_forest", "xgboost", "lightgbm", "neural_network", "ensemble"],
        help="Models to evaluate"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    if args.verbose:
        logger.info(f"Configuration loaded from {args.config}")
        logger.info(f"Output directory: {args.output}")
        logger.info(f"Models to evaluate: {args.models}")
    
    try:
        # Evaluate models
        comparison_df, results_list, model_names = evaluate_multiple_models(config)
        
        # Generate report
        generate_evaluation_report(comparison_df, results_list, model_names, args.output)
        
        # Create visualizations
        create_visualizations(comparison_df, results_list, model_names, args.output)
        
        # Print summary
        logger.info("Evaluation Summary:")
        logger.info(f"  Models evaluated: {len(model_names)}")
        logger.info(f"  Best AUC-ROC: {comparison_df.loc[comparison_df['AUC-ROC'].idxmax(), 'Model']}")
        logger.info(f"  Best F1-Score: {comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Model']}")
        logger.info(f"  Results saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
