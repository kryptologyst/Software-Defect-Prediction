"""Feature engineering for software defect prediction."""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from loguru import logger


class FeatureEngineer:
    """Feature engineering pipeline for software defect prediction."""
    
    def __init__(self, scaler_type: str = "standard"):
        """Initialize feature engineer.
        
        Args:
            scaler_type: Type of scaler to use ("standard" or "robust")
        """
        self.scaler_type = scaler_type
        self.scaler = StandardScaler() if scaler_type == "standard" else RobustScaler()
        self.feature_selector = None
        self.selected_features = None
        self.is_fitted = False
        
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between code metrics.
        
        Args:
            df: DataFrame with basic features
            
        Returns:
            DataFrame with additional interaction features
        """
        df_enhanced = df.copy()
        
        # Complexity per function
        df_enhanced['complexity_per_function'] = (
            df['cyclomatic_complexity'] / df['num_functions']
        )
        
        # Lines per class
        df_enhanced['lines_per_class'] = (
            df['lines_of_code'] / (df['num_classes'] + 1)  # +1 to avoid division by zero
        )
        
        # Function density
        df_enhanced['function_density'] = (
            df['num_functions'] / df['lines_of_code']
        )
        
        # Import density
        df_enhanced['import_density'] = (
            df['num_imports'] / df['lines_of_code']
        )
        
        # Risk score based on multiple factors
        df_enhanced['risk_score'] = (
            df['cyclomatic_complexity'] * 0.3 +
            df['max_nesting_depth'] * 0.2 +
            (1 - df['test_coverage']) * 0.3 +
            df['duplicate_lines'] * 0.1 +
            (1 - df['comment_density']) * 0.1
        )
        
        # Size complexity interaction
        df_enhanced['size_complexity_interaction'] = (
            df['lines_of_code'] * df['cyclomatic_complexity']
        )
        
        # Coverage quality score
        df_enhanced['coverage_quality'] = (
            df['test_coverage'] * df['comment_density']
        )
        
        return df_enhanced
    
    def create_polynomial_features(self, df: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
        """Create polynomial features for non-linear relationships.
        
        Args:
            df: DataFrame with features
            degree: Degree of polynomial features
            
        Returns:
            DataFrame with polynomial features
        """
        df_poly = df.copy()
        
        # Create polynomial features for key metrics
        key_features = ['cyclomatic_complexity', 'lines_of_code', 'avg_function_length']
        
        for feature in key_features:
            if feature in df.columns:
                for d in range(2, degree + 1):
                    df_poly[f'{feature}_power_{d}'] = df[feature] ** d
        
        return df_poly
    
    def create_binned_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binned features for categorical analysis.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with binned features
        """
        df_binned = df.copy()
        
        # Bin cyclomatic complexity
        df_binned['complexity_level'] = pd.cut(
            df['cyclomatic_complexity'],
            bins=[0, 5, 10, 15, 30],
            labels=['low', 'medium', 'high', 'very_high']
        )
        
        # Bin lines of code
        df_binned['size_level'] = pd.cut(
            df['lines_of_code'],
            bins=[0, 100, 300, 500, 2000],
            labels=['small', 'medium', 'large', 'very_large']
        )
        
        # Bin test coverage
        df_binned['coverage_level'] = pd.cut(
            df['test_coverage'],
            bins=[0, 0.3, 0.6, 0.8, 1.0],
            labels=['low', 'medium', 'high', 'excellent']
        )
        
        return df_binned
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Fit feature engineering pipeline and transform data.
        
        Args:
            X: Feature matrix
            y: Target labels (optional, for feature selection)
            
        Returns:
            Transformed feature matrix
        """
        logger.info("Starting feature engineering pipeline")
        
        # Create interaction features
        X_enhanced = self.create_interaction_features(X)
        logger.info(f"Created interaction features: {X_enhanced.shape[1]} features")
        
        # Create polynomial features
        X_enhanced = self.create_polynomial_features(X_enhanced, degree=2)
        logger.info(f"Created polynomial features: {X_enhanced.shape[1]} features")
        
        # Create binned features
        X_enhanced = self.create_binned_features(X_enhanced)
        logger.info(f"Created binned features: {X_enhanced.shape[1]} features")
        
        # Handle categorical features
        X_processed = self._handle_categorical_features(X_enhanced)
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_processed),
            columns=X_processed.columns,
            index=X_processed.index
        )
        
        # Feature selection if target is provided
        if y is not None:
            self.feature_selector = SelectKBest(score_func=f_classif, k=20)
            X_selected = self.feature_selector.fit_transform(X_scaled, y)
            self.selected_features = X_scaled.columns[self.feature_selector.get_support()]
            
            X_final = pd.DataFrame(
                X_selected,
                columns=self.selected_features,
                index=X_scaled.index
            )
            logger.info(f"Selected {len(self.selected_features)} best features")
        else:
            X_final = X_scaled
        
        self.is_fitted = True
        logger.info(f"Feature engineering complete: {X_final.shape[1]} final features")
        
        return X_final
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted pipeline.
        
        Args:
            X: Feature matrix
            
        Returns:
            Transformed feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Feature engineer must be fitted before transform")
        
        # Apply same transformations
        X_enhanced = self.create_interaction_features(X)
        X_enhanced = self.create_polynomial_features(X_enhanced, degree=2)
        X_enhanced = self.create_binned_features(X_enhanced)
        X_processed = self._handle_categorical_features(X_enhanced)
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_processed),
            columns=X_processed.columns,
            index=X_processed.index
        )
        
        # Apply feature selection
        if self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X_scaled)
            X_final = pd.DataFrame(
                X_selected,
                columns=self.selected_features,
                index=X_scaled.index
            )
        else:
            X_final = X_scaled
        
        return X_final
    
    def _handle_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle categorical features by encoding them.
        
        Args:
            df: DataFrame with categorical features
            
        Returns:
            DataFrame with encoded categorical features
        """
        df_encoded = df.copy()
        
        # One-hot encode categorical features
        categorical_cols = df_encoded.select_dtypes(include=['category', 'object']).columns
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                dummies = pd.get_dummies(df_encoded[col], prefix=col)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                df_encoded = df_encoded.drop(col, axis=1)
        
        return df_encoded
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get feature importance scores from fitted selector.
        
        Returns:
            Series with feature importance scores
        """
        if self.feature_selector is None or self.selected_features is None:
            return None
        
        scores = self.feature_selector.scores_
        importance = pd.Series(scores, index=self.selected_features)
        return importance.sort_values(ascending=False)


def create_feature_pipeline(
    scaler_type: str = "standard",
    feature_selection: bool = True,
    k_best: int = 20
) -> FeatureEngineer:
    """Create a feature engineering pipeline.
    
    Args:
        scaler_type: Type of scaler to use
        feature_selection: Whether to perform feature selection
        k_best: Number of best features to select
        
    Returns:
        Configured FeatureEngineer instance
    """
    engineer = FeatureEngineer(scaler_type=scaler_type)
    return engineer


if __name__ == "__main__":
    # Test feature engineering
    from .synthetic_data import generate_synthetic_dataset
    
    # Generate sample data
    data = generate_synthetic_dataset(n_samples=100)
    
    # Create feature engineer
    engineer = FeatureEngineer()
    
    # Fit and transform
    X_processed = engineer.fit_transform(data['X'], data['y'])
    
    print(f"Original features: {data['X'].shape[1]}")
    print(f"Processed features: {X_processed.shape[1]}")
    print(f"Feature names: {list(X_processed.columns)}")
    
    if engineer.get_feature_importance() is not None:
        print("\nTop 10 features by importance:")
        print(engineer.get_feature_importance().head(10))
