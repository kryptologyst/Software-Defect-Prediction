#!/usr/bin/env python3
"""Training script for software defect prediction models."""

import argparse
import sys
import os
from pathlib import Path
from typing import Dict, Any
import yaml
from loguru import logger

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.synthetic_data import generate_synthetic_dataset, create_train_test_split
from models.defect_predictor import DefectPredictor
from evaluation.metrics import DefectPredictionEvaluator
from explainability.shap_explainer import DefectPredictionExplainer


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


def setup_logging(config: Dict[str, Any]) -> None:
    """Setup logging configuration.
    
    Args:
        config: Configuration dictionary
    """
    log_config = config.get('logging', {})
    
    # Create logs directory
    log_file = log_config.get('file', 'logs/training.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logger
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        level=log_config.get('level', 'INFO'),
        format=log_config.get('format', '{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}')
    )
    logger.add(
        log_file,
        level=log_config.get('level', 'INFO'),
        format=log_config.get('format', '{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}')
    )


def train_model(config: Dict[str, Any]) -> Dict[str, Any]:
    """Train defect prediction model.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Training results dictionary
    """
    logger.info("Starting model training")
    
    # Extract configuration
    data_config = config['data']
    model_config = config['model']
    features_config = config.get('features', {})
    
    # Generate dataset
    logger.info(f"Generating dataset with {data_config['n_samples']} samples")
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
    
    logger.info(f"Training set: {len(split_data['X_train'])} samples")
    logger.info(f"Test set: {len(split_data['X_test'])} samples")
    logger.info(f"Defect rate: {split_data['y_train'].mean():.1%}")
    
    # Initialize model
    predictor = DefectPredictor(
        model_type=model_config['name'],
        random_state=model_config['random_state']
    )
    
    # Train model
    logger.info(f"Training {model_config['name']} model")
    predictor.fit(
        split_data['X_train'],
        split_data['y_train'],
        use_feature_engineering=features_config.get('feature_selection', True)
    )
    
    # Evaluate model
    logger.info("Evaluating model")
    evaluator = DefectPredictionEvaluator(random_state=data_config['random_state'])
    
    # Training evaluation
    train_results = evaluator.evaluate_model(
        predictor.model,
        split_data['X_train'],
        split_data['y_train'],
        cv_folds=config.get('evaluation', {}).get('cv_folds', 5)
    )
    
    # Test evaluation
    test_results = evaluator.evaluate_model(
        predictor.model,
        split_data['X_test'],
        split_data['y_test'],
        cv_folds=config.get('evaluation', {}).get('cv_folds', 5)
    )
    
    # Generate explanations
    logger.info("Generating SHAP explanations")
    explainer = DefectPredictionExplainer(
        predictor.model,
        split_data['feature_names'],
        random_state=data_config['random_state']
    )
    
    shap_results = explainer.explain_with_shap(
        split_data['X_test'],
        sample_size=config.get('explainability', {}).get('shap_sample_size', 100)
    )
    
    # Compile results
    results = {
        'config': config,
        'data': data,
        'split_data': split_data,
        'predictor': predictor,
        'train_results': train_results,
        'test_results': test_results,
        'explainer': explainer,
        'shap_results': shap_results
    }
    
    logger.info("Training completed successfully")
    return results


def save_results(results: Dict[str, Any], output_dir: str) -> None:
    """Save training results.
    
    Args:
        results: Training results dictionary
        output_dir: Output directory path
    """
    logger.info(f"Saving results to {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, 'model.pkl')
    results['predictor'].model.save_model(model_path)
    
    # Save predictions
    test_pred = results['predictor'].predict(results['split_data']['X_test'])
    test_proba = results['predictor'].predict_proba(results['split_data']['X_test'])
    
    predictions_df = results['split_data']['X_test'].copy()
    predictions_df['actual'] = results['split_data']['y_test']
    predictions_df['predicted'] = test_pred
    predictions_df['defect_probability'] = test_proba[:, 1]
    
    predictions_path = os.path.join(output_dir, 'predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    
    # Save explanations
    explainer = results['explainer']
    explanations_path = os.path.join(output_dir, 'explanations.pkl')
    explainer.save_explanations(results['split_data']['X_test'], explanations_path)
    
    # Save evaluation report
    evaluator = DefectPredictionEvaluator()
    train_report = evaluator.create_evaluation_report(
        results['train_results'], 
        f"{results['config']['model']['name']} (Train)"
    )
    test_report = evaluator.create_evaluation_report(
        results['test_results'], 
        f"{results['config']['model']['name']} (Test)"
    )
    
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write("TRAINING RESULTS\n")
        f.write("=" * 50 + "\n")
        f.write(train_report)
        f.write("\n\nTEST RESULTS\n")
        f.write("=" * 50 + "\n")
        f.write(test_report)
    
    # Save configuration
    config_path = os.path.join(output_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(results['config'], f, default_flow_style=False)
    
    logger.info(f"Results saved to {output_dir}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train software defect prediction model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="results",
        help="Output directory for results"
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
    setup_logging(config)
    
    if args.verbose:
        logger.info(f"Configuration loaded from {args.config}")
        logger.info(f"Output directory: {args.output}")
    
    try:
        # Train model
        results = train_model(config)
        
        # Save results
        save_results(results, args.output)
        
        # Print summary
        test_results = results['test_results']
        logger.info("Training Summary:")
        logger.info(f"  Model: {config['model']['name']}")
        logger.info(f"  Test Accuracy: {test_results['metrics'].accuracy:.3f}")
        logger.info(f"  Test AUC-ROC: {test_results['metrics'].auc_roc:.3f}")
        logger.info(f"  Test Precision: {test_results['metrics'].precision:.3f}")
        logger.info(f"  Test Recall: {test_results['metrics'].recall:.3f}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
