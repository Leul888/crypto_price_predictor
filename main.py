"""
Application Entry Point for Cryptocurrency Prediction

This script sets up and runs the entire application,
including data collection, feature engineering, and model training.
"""

import sys
sys.path.append('.')

from data_collector import CryptoDataCollector
from feature_engineering import FeatureEngineer
from ml_models import CryptoPricePredictor, get_hyperparameter_grids
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)


def main():
    """Run the application pipeline."""
    logger = logging.getLogger(__name__)
    
    # Initialize components
    collector = CryptoDataCollector()
    engineer = FeatureEngineer()
    predictor = CryptoPricePredictor()
    
    # Collect data
    logger.info("Collecting data...")
    btc_data = collector.fetch_crypto_data('BTC', period='6mo', interval='1h')
    
    # Feature engineering
    logger.info("Engineering features...")
    featured_data = engineer.create_all_features(btc_data, target_type='price', prediction_horizon=1)
    
    # Train all models
    logger.info("Training models...")
    results = predictor.train_all_models(featured_data)
    logger.info("Model Performance Comparison:")
    logger.info(results[['model_name', 'test_rmse', 'test_r2']])
    
    # Hyperparameter tuning demonstration
    logger.info("Performing hyperparameter tuning...")
    param_grids = get_hyperparameter_grids()
    X, y, feature_names = predictor.prepare_features(featured_data)
    best_params = predictor.hyperparameter_tuning('random_forest', X, y, param_grids['random_forest'])
    logger.info(f"Best Parameters for Random Forest: {best_params}")
    
    # Save the best model
    logger.info("Saving best model...")
    best_model_name = results.iloc[0]['model_name']
    predictor.save_model(best_model_name, 'BTC')
    

if __name__ == "__main__":
    main()
