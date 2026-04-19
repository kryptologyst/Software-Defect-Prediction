"""Tests for software defect prediction package."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.synthetic_data import generate_synthetic_dataset, SyntheticDataGenerator, DatasetConfig
from models.defect_predictor import DefectPredictor, RandomForestDefectPredictor
from evaluation.metrics import DefectPredictionEvaluator, EvaluationMetrics
from features.feature_engineering import FeatureEngineer


class TestSyntheticData:
    """Test synthetic data generation."""
    
    def test_generate_synthetic_dataset(self):
        """Test synthetic dataset generation."""
        data = generate_synthetic_dataset(n_samples=100, random_state=42)
        
        assert 'features' in data
        assert 'X' in data
        assert 'y' in data
        assert 'feature_names' in data
        
        assert len(data['features']) == 100
        assert len(data['X']) == 100
        assert len(data['y']) == 100
        assert len(data['feature_names']) > 0
        
        # Check feature names
        expected_features = [
            'lines_of_code', 'cyclomatic_complexity', 'num_functions',
            'num_classes', 'num_imports', 'avg_function_length',
            'max_nesting_depth', 'comment_density', 'duplicate_lines',
            'test_coverage'
        ]
        
        for feature in expected_features:
            assert feature in data['feature_names']
    
    def test_dataset_config(self):
        """Test dataset configuration."""
        config = DatasetConfig(n_samples=50, defect_ratio=0.2)
        
        assert config.n_samples == 50
        assert config.defect_ratio == 0.2
        assert config.random_state == 42
    
    def test_data_generator(self):
        """Test synthetic data generator."""
        config = DatasetConfig(n_samples=50, random_state=42)
        generator = SyntheticDataGenerator(config)
        
        dataset = generator.generate_dataset()
        
        assert len(dataset['features']) == 50
        assert len(dataset['y']) == 50
        assert dataset['y'].sum() > 0  # Should have some defects


class TestDefectPredictor:
    """Test defect prediction models."""
    
    def test_random_forest_predictor(self):
        """Test Random Forest predictor."""
        predictor = RandomForestDefectPredictor(random_state=42)
        
        # Generate test data
        data = generate_synthetic_dataset(n_samples=100, random_state=42)
        
        # Train model
        predictor.fit(data['X'], data['y'])
        
        # Make predictions
        predictions = predictor.predict(data['X'])
        probabilities = predictor.predict_proba(data['X'])
        
        assert len(predictions) == 100
        assert len(probabilities) == 100
        assert probabilities.shape[1] == 2  # Binary classification
        
        # Check feature importance
        importance = predictor.get_feature_importance()
        assert len(importance) > 0
        assert all(importance >= 0)  # Importance should be non-negative
    
    def test_defect_predictor_interface(self):
        """Test main defect predictor interface."""
        predictor = DefectPredictor(model_type="random_forest", random_state=42)
        
        # Generate test data
        data = generate_synthetic_dataset(n_samples=100, random_state=42)
        
        # Train model
        predictor.fit(data['X'], data['y'])
        
        # Make predictions
        predictions = predictor.predict(data['X'])
        probabilities = predictor.predict_proba(data['X'])
        
        assert len(predictions) == 100
        assert len(probabilities) == 100
    
    def test_ensemble_predictor(self):
        """Test ensemble predictor."""
        predictor = DefectPredictor(model_type="ensemble", random_state=42)
        
        # Generate test data
        data = generate_synthetic_dataset(n_samples=100, random_state=42)
        
        # Train model
        predictor.fit(data['X'], data['y'])
        
        # Make predictions
        predictions = predictor.predict(data['X'])
        probabilities = predictor.predict_proba(data['X'])
        
        assert len(predictions) == 100
        assert len(probabilities) == 100


class TestEvaluation:
    """Test evaluation metrics."""
    
    def test_evaluation_metrics(self):
        """Test evaluation metrics calculation."""
        evaluator = DefectPredictionEvaluator(random_state=42)
        
        # Create mock data
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1])
        y_proba = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.6, 0.4], [0.2, 0.8]])
        
        metrics = evaluator.calculate_basic_metrics(y_true, y_pred, y_proba)
        
        assert isinstance(metrics, EvaluationMetrics)
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1
        assert 0 <= metrics.f1 <= 1
        assert 0 <= metrics.auc_roc <= 1
        assert 0 <= metrics.auc_pr <= 1
    
    def test_evaluator(self):
        """Test evaluator with model."""
        evaluator = DefectPredictionEvaluator(random_state=42)
        
        # Create mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0, 1, 0, 1])
        mock_model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.2, 0.8]])
        
        # Mock cross-validation
        with patch('sklearn.model_selection.cross_val_score') as mock_cv:
            mock_cv.return_value = np.array([0.8, 0.85, 0.82, 0.88, 0.86])
            
            X = pd.DataFrame(np.random.randn(4, 5))
            y = np.array([0, 1, 0, 1])
            
            results = evaluator.evaluate_model(mock_model, X, y)
            
            assert 'cv_scores' in results
            assert 'metrics' in results
            assert 'predictions' in results
            assert 'probabilities' in results


class TestFeatureEngineering:
    """Test feature engineering."""
    
    def test_feature_engineer(self):
        """Test feature engineering pipeline."""
        engineer = FeatureEngineer(scaler_type="standard")
        
        # Generate test data
        data = generate_synthetic_dataset(n_samples=100, random_state=42)
        
        # Fit and transform
        X_processed = engineer.fit_transform(data['X'], data['y'])
        
        assert X_processed.shape[0] == 100
        assert X_processed.shape[1] > data['X'].shape[1]  # Should have more features
        
        # Test transform on new data
        X_new = data['X'].iloc[:10]
        X_transformed = engineer.transform(X_new)
        
        assert X_transformed.shape[0] == 10
        assert X_transformed.shape[1] == X_processed.shape[1]
    
    def test_interaction_features(self):
        """Test interaction feature creation."""
        engineer = FeatureEngineer()
        
        # Create simple test data
        X = pd.DataFrame({
            'lines_of_code': [100, 200, 300],
            'cyclomatic_complexity': [5, 10, 15],
            'num_functions': [5, 10, 15]
        })
        
        X_enhanced = engineer.create_interaction_features(X)
        
        # Should have additional features
        assert X_enhanced.shape[1] > X.shape[1]
        assert 'complexity_per_function' in X_enhanced.columns
        assert 'function_density' in X_enhanced.columns


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline."""
        # Generate data
        data = generate_synthetic_dataset(n_samples=200, random_state=42)
        
        # Create predictor
        predictor = DefectPredictor(model_type="random_forest", random_state=42)
        
        # Train with feature engineering
        predictor.fit(data['X'], data['y'], use_feature_engineering=True)
        
        # Make predictions
        predictions = predictor.predict(data['X'])
        probabilities = predictor.predict_proba(data['X'])
        
        # Evaluate
        evaluator = DefectPredictionEvaluator(random_state=42)
        results = evaluator.evaluate_model(predictor.model, data['X'], data['y'])
        
        # Check results
        assert len(predictions) == 200
        assert len(probabilities) == 200
        assert results['metrics'].accuracy > 0.5  # Should be better than random
        assert results['metrics'].auc_roc > 0.5  # Should be better than random


if __name__ == "__main__":
    pytest.main([__file__])
