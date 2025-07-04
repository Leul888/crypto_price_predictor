"""
Machine Learning Models for Cryptocurrency Price Prediction

This module contains various ML models and evaluation utilities for
cryptocurrency price prediction.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import os
import logging
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class CryptoPricePredictor:
    """Advanced cryptocurrency price prediction using multiple ML models."""
    
    def __init__(self, model_dir: str = 'models'):
        self.model_dir = model_dir
        self.scalers = {}
        self.models = {}
        self.model_scores = {}
        
        # Create models directory
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize available models
        self.available_models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100, 
                random_state=42, 
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100, 
                random_state=42
            ),
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0),
            'svr': SVR(kernel='rbf', C=1.0),
            'neural_network': MLPRegressor(
                hidden_layer_sizes=(100, 50), 
                random_state=42, 
                max_iter=500
            )
        }
    
    def prepare_features(self, df: pd.DataFrame, 
                        target_col: str = 'target',
                        feature_cols: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features and target for model training.
        
        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            feature_cols: List of feature columns to use (if None, use all except target and datetime)
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        df = df.copy()
        
        # Remove non-numeric and unwanted columns
        exclude_cols = ['datetime', 'symbol', target_col]
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            # Only keep numeric columns
            feature_cols = [col for col in feature_cols if df[col].dtype in ['int64', 'float64']]
        
        # Handle infinite values
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        
        # Drop rows with NaN values
        df = df.dropna(subset=feature_cols + [target_col])
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        logger.info(f"Prepared features: {X.shape}, Target: {y.shape}")
        return X, y, feature_cols
    
    def scale_features(self, X_train: np.ndarray, X_test: np.ndarray, 
                      scaler_type: str = 'standard') -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale features using specified scaler.
        
        Args:
            X_train: Training features
            X_test: Test features
            scaler_type: Type of scaler ('standard' or 'minmax')
            
        Returns:
            Tuple of scaled (X_train, X_test)
        """
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Store scaler for later use
        self.scalers[scaler_type] = scaler
        
        return X_train_scaled, X_test_scaled
    
    def train_single_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray,
                          scale_features: bool = False) -> Dict[str, Any]:
        """
        Train a single model and evaluate its performance.
        
        Args:
            model_name: Name of the model to train
            X_train, y_train: Training data
            X_test, y_test: Test data
            scale_features: Whether to scale features
            
        Returns:
            Dictionary with model performance metrics
        """
        if model_name not in self.available_models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = self.available_models[model_name]
        
        # Scale features if required
        if scale_features and model_name in ['svr', 'neural_network', 'linear_regression', 'ridge', 'lasso']:
            X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        else:
            X_train_scaled, X_test_scaled = X_train, X_test
        
        # Train model
        logger.info(f"Training {model_name}...")
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'model_name': model_name,
            'train_mse': mean_squared_error(y_train, y_pred_train),
            'test_mse': mean_squared_error(y_test, y_pred_test),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test))
        }
        
        # Store model
        self.models[model_name] = model
        self.model_scores[model_name] = metrics
        
        logger.info(f"{model_name} - Test RMSE: {metrics['test_rmse']:.4f}, Test R²: {metrics['test_r2']:.4f}")
        
        return metrics
    
    def train_all_models(self, df: pd.DataFrame, target_col: str = 'target',
                        test_size: float = 0.2, random_state: int = 42) -> pd.DataFrame:
        """
        Train all available models and compare their performance.
        
        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            test_size: Proportion of data for testing
            random_state: Random state for reproducibility
            
        Returns:
            DataFrame with model comparison results
        """
        # Prepare data
        X, y, feature_names = self.prepare_features(df, target_col)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Test data shape: {X_test.shape}")
        
        # Train all models
        results = []
        for model_name in self.available_models.keys():
            try:
                metrics = self.train_single_model(
                    model_name, X_train, y_train, X_test, y_test, 
                    scale_features=True
                )
                results.append(metrics)
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('test_rmse')
        
        return results_df
    
    def hyperparameter_tuning(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                             param_grid: Dict[str, List], cv: int = 5) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning for a specific model.
        
        Args:
            model_name: Name of the model
            X_train, y_train: Training data
            param_grid: Parameter grid for tuning
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with best parameters and score
        """
        if model_name not in self.available_models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = self.available_models[model_name]
        
        logger.info(f"Hyperparameter tuning for {model_name}...")
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        results = {
            'model_name': model_name,
            'best_params': grid_search.best_params_,
            'best_score': -grid_search.best_score_,
            'best_model': grid_search.best_estimator_
        }
        
        # Update stored model with best parameters
        self.models[model_name] = grid_search.best_estimator_
        
        logger.info(f"Best parameters for {model_name}: {results['best_params']}")
        logger.info(f"Best CV score: {results['best_score']:.4f}")
        
        return results
    
    def get_feature_importance(self, model_name: str, feature_names: List[str]) -> pd.DataFrame:
        """
        Get feature importance for tree-based models.
        
        Args:
            model_name: Name of the model
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance scores
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            logger.warning(f"Model {model_name} does not have feature importances")
            return pd.DataFrame()
    
    def make_predictions(self, model_name: str, X: np.ndarray, 
                        scale_features: bool = False) -> np.ndarray:
        """
        Make predictions using a trained model.
        
        Args:
            model_name: Name of the model
            X: Input features
            scale_features: Whether to scale features
            
        Returns:
            Predictions array
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.models[model_name]
        
        # Scale features if required
        if scale_features and 'standard' in self.scalers:
            X_scaled = self.scalers['standard'].transform(X)
        else:
            X_scaled = X
        
        predictions = model.predict(X_scaled)
        return predictions
    
    def save_model(self, model_name: str, symbol: str) -> None:
        """
        Save a trained model and its scaler.
        
        Args:
            model_name: Name of the model
            symbol: Cryptocurrency symbol
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        # Save model
        model_filename = os.path.join(self.model_dir, f'{symbol}_{model_name}_model.pkl')
        joblib.dump(self.models[model_name], model_filename)
        
        # Save scaler if exists
        if 'standard' in self.scalers:
            scaler_filename = os.path.join(self.model_dir, f'{symbol}_{model_name}_scaler.pkl')
            joblib.dump(self.scalers['standard'], scaler_filename)
        
        logger.info(f"Model {model_name} saved for {symbol}")
    
    def load_model(self, model_name: str, symbol: str) -> None:
        """
        Load a saved model and its scaler.
        
        Args:
            model_name: Name of the model
            symbol: Cryptocurrency symbol
        """
        # Load model
        model_filename = os.path.join(self.model_dir, f'{symbol}_{model_name}_model.pkl')
        if os.path.exists(model_filename):
            self.models[model_name] = joblib.load(model_filename)
        else:
            raise FileNotFoundError(f"Model file not found: {model_filename}")
        
        # Load scaler if exists
        scaler_filename = os.path.join(self.model_dir, f'{symbol}_{model_name}_scaler.pkl')
        if os.path.exists(scaler_filename):
            self.scalers['standard'] = joblib.load(scaler_filename)
        
        logger.info(f"Model {model_name} loaded for {symbol}")
    
    def evaluate_model_performance(self, model_name: str, X_test: np.ndarray, 
                                 y_test: np.ndarray, scale_features: bool = False) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            model_name: Name of the model
            X_test, y_test: Test data
            scale_features: Whether to scale features
            
        Returns:
            Dictionary with performance metrics
        """
        predictions = self.make_predictions(model_name, X_test, scale_features)
        
        metrics = {
            'mse': mean_squared_error(y_test, predictions),
            'mae': mean_absolute_error(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'r2': r2_score(y_test, predictions)
        }
        
        return metrics


def get_hyperparameter_grids() -> Dict[str, Dict[str, List]]:
    """
    Get hyperparameter grids for tuning different models.
    
    Returns:
        Dictionary with parameter grids for each model
    """
    param_grids = {
        'random_forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        },
        'gradient_boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        },
        'ridge': {
            'alpha': [0.1, 1.0, 10.0, 100.0]
        },
        'lasso': {
            'alpha': [0.1, 1.0, 10.0, 100.0]
        },
        'svr': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.01, 0.1]
        },
        'neural_network': {
            'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100)],
            'alpha': [0.0001, 0.001, 0.01]
        }
    }
    
    return param_grids


def main():
    """Example usage of CryptoPricePredictor."""
    # This would typically use real data from feature engineering
    import sys
    sys.path.append('.')
    
    # Create sample data for demonstration
    from feature_engineering import FeatureEngineer
    import pandas as pd
    import numpy as np
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='H')
    close_prices = 100 + np.random.randn(1000).cumsum() * 0.5
    
    sample_data = pd.DataFrame({
        'datetime': dates,
        'open': close_prices * (1 + np.random.randn(1000) * 0.001),
        'high': close_prices * (1 + np.abs(np.random.randn(1000)) * 0.002),
        'low': close_prices * (1 - np.abs(np.random.randn(1000)) * 0.002),
        'close': close_prices,
        'volume': np.random.randint(1000, 10000, 1000),
        'symbol': 'BTC'
    })
    
    # Create features
    fe = FeatureEngineer()
    featured_data = fe.create_all_features(sample_data, target_type='price', prediction_horizon=1)
    
    # Train models
    predictor = CryptoPricePredictor()
    results = predictor.train_all_models(featured_data)
    
    print("Model Performance Comparison:")
    print(results[['model_name', 'test_rmse', 'test_r2']].head())
    
    # Get feature importance for best model
    best_model = results.iloc[0]['model_name']
    X, y, feature_names = predictor.prepare_features(featured_data)
    
    try:
        importance = predictor.get_feature_importance(best_model, feature_names)
        print(f"\nTop 10 features for {best_model}:")
        print(importance.head(10))
    except:
        print(f"Feature importance not available for {best_model}")


if __name__ == "__main__":
    main()
