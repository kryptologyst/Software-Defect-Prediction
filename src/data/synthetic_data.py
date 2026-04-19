"""Data processing and synthetic dataset generation for software defect prediction."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import random
from loguru import logger


@dataclass
class DatasetConfig:
    """Configuration for synthetic dataset generation."""
    n_samples: int = 1000
    n_features: int = 10
    defect_ratio: float = 0.3
    random_state: int = 42
    noise_level: float = 0.1


class SyntheticDataGenerator:
    """Generate synthetic software defect prediction datasets."""
    
    def __init__(self, config: DatasetConfig):
        """Initialize the data generator.
        
        Args:
            config: Dataset configuration parameters
        """
        self.config = config
        self._set_seed()
        
    def _set_seed(self) -> None:
        """Set random seeds for reproducibility."""
        random.seed(self.config.random_state)
        np.random.seed(self.config.random_state)
        
    def generate_features(self) -> pd.DataFrame:
        """Generate synthetic code metrics features.
        
        Returns:
            DataFrame with synthetic code metrics
        """
        n_samples = self.config.n_samples
        
        # Generate correlated features based on realistic software metrics
        data = {}
        
        # Lines of code (base feature)
        data['lines_of_code'] = np.random.lognormal(mean=5.5, sigma=0.8, size=n_samples).astype(int)
        data['lines_of_code'] = np.clip(data['lines_of_code'], 10, 2000)
        
        # Cyclomatic complexity (correlated with LOC)
        complexity_base = np.log(data['lines_of_code']) * 2 + np.random.normal(0, 1, n_samples)
        data['cyclomatic_complexity'] = np.clip(complexity_base, 1, 30).astype(int)
        
        # Number of functions (correlated with LOC)
        func_base = data['lines_of_code'] / 20 + np.random.normal(0, 2, n_samples)
        data['num_functions'] = np.clip(func_base, 1, 50).astype(int)
        
        # Number of classes
        class_base = data['lines_of_code'] / 100 + np.random.normal(0, 1, n_samples)
        data['num_classes'] = np.clip(class_base, 0, 20).astype(int)
        
        # Number of imports
        import_base = data['lines_of_code'] / 50 + np.random.normal(0, 1, n_samples)
        data['num_imports'] = np.clip(import_base, 0, 30).astype(int)
        
        # Average function length
        data['avg_function_length'] = data['lines_of_code'] / data['num_functions']
        
        # Maximum nesting depth (correlated with complexity)
        nesting_base = np.log(data['cyclomatic_complexity']) + np.random.normal(0, 0.5, n_samples)
        data['max_nesting_depth'] = np.clip(nesting_base, 1, 8).astype(int)
        
        # Comment density (inversely correlated with defects)
        comment_base = np.random.beta(2, 5, n_samples) * 0.3
        data['comment_density'] = np.clip(comment_base, 0.01, 0.5)
        
        # Duplicate lines (risk factor)
        dup_base = np.random.exponential(0.1, n_samples) * data['lines_of_code'] / 100
        data['duplicate_lines'] = np.clip(dup_base, 0, 50).astype(int)
        
        # Test coverage (inversely correlated with defects)
        coverage_base = np.random.beta(3, 2, n_samples)
        data['test_coverage'] = np.clip(coverage_base, 0.1, 1.0)
        
        return pd.DataFrame(data)
    
    def generate_labels(self, features: pd.DataFrame) -> np.ndarray:
        """Generate defect labels based on feature patterns.
        
        Args:
            features: DataFrame with code metrics
            
        Returns:
            Binary defect labels (1 = defect-prone, 0 = clean)
        """
        n_samples = len(features)
        
        # Create risk score based on multiple factors
        risk_score = np.zeros(n_samples)
        
        # High complexity increases risk
        risk_score += features['cyclomatic_complexity'] / 30 * 0.3
        
        # Large functions increase risk
        risk_score += (features['avg_function_length'] > 30).astype(int) * 0.2
        
        # Deep nesting increases risk
        risk_score += features['max_nesting_depth'] / 8 * 0.2
        
        # Low test coverage increases risk
        risk_score += (1 - features['test_coverage']) * 0.2
        
        # High duplicate lines increase risk
        risk_score += np.clip(features['duplicate_lines'] / 20, 0, 1) * 0.1
        
        # Add noise
        risk_score += np.random.normal(0, self.config.noise_level, n_samples)
        
        # Convert to binary labels
        threshold = np.percentile(risk_score, (1 - self.config.defect_ratio) * 100)
        labels = (risk_score > threshold).astype(int)
        
        return labels
    
    def generate_dataset(self) -> Dict[str, pd.DataFrame]:
        """Generate complete synthetic dataset.
        
        Returns:
            Dictionary containing features and labels
        """
        logger.info(f"Generating synthetic dataset with {self.config.n_samples} samples")
        
        # Generate features
        features = self.generate_features()
        
        # Generate labels
        labels = self.generate_labels(features)
        
        # Add labels to features
        features['defect'] = labels
        
        logger.info(f"Generated dataset: {len(features)} samples, {labels.sum()} defects ({labels.mean():.1%})")
        
        return {
            'features': features,
            'X': features.drop('defect', axis=1),
            'y': labels,
            'feature_names': list(features.columns[:-1])
        }


def generate_synthetic_dataset(
    n_samples: int = 1000,
    defect_ratio: float = 0.3,
    random_state: int = 42
) -> Dict[str, pd.DataFrame]:
    """Convenience function to generate synthetic dataset.
    
    Args:
        n_samples: Number of samples to generate
        defect_ratio: Ratio of defect-prone samples
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing features and labels
    """
    config = DatasetConfig(
        n_samples=n_samples,
        defect_ratio=defect_ratio,
        random_state=random_state
    )
    
    generator = SyntheticDataGenerator(config)
    return generator.generate_dataset()


def create_train_test_split(
    data: Dict[str, pd.DataFrame],
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, pd.DataFrame]:
    """Create train-test split with stratification.
    
    Args:
        data: Dataset dictionary
        test_size: Proportion of data for testing
        random_state: Random seed
        
    Returns:
        Dictionary with train/test splits
    """
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        data['X'], data['y'],
        test_size=test_size,
        random_state=random_state,
        stratify=data['y']
    )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': data['feature_names']
    }


if __name__ == "__main__":
    # Generate and save sample dataset
    data = generate_synthetic_dataset(n_samples=1000)
    
    # Save to CSV
    data['features'].to_csv('data/synthetic_defect_dataset.csv', index=False)
    logger.info("Saved synthetic dataset to data/synthetic_defect_dataset.csv")
    
    # Print dataset info
    print(f"Dataset shape: {data['features'].shape}")
    print(f"Defect ratio: {data['y'].mean():.1%}")
    print(f"Features: {data['feature_names']}")
