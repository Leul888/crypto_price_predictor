"""
Cryptocurrency Price Predictor

This module uses machine learning to predict short-term price movements
of cryptocurrencies.
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import pickle
import os


class CryptoPricePredictor:
    """Predicts cryptocurrency price movements."""
    
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for the model.

        Args:
            data: DataFrame with cryptocurrency data

        Returns:
            DataFrame with feature engineering applied
        """
        # Feature engineering example
        data['hour'] = data['datetime'].dt.hour
        data['day'] = data['datetime'].dt.day
        data['month'] = data['datetime'].dt.month

        # Lag features
        for lag in range(1, 4):
            data[f'close_lag_{lag}'] = data['close'].shift(lag)

        # Drop NaNs
        data = data.dropna()
        return data

    def train_model(self, data: pd.DataFrame, symbol: str):
        """
        Train the machine learning model.

        Args:
            data: DataFrame with features and target
            symbol: Cryptocurrency symbol for which to train the model
        """
        data = self._prepare_features(data)
        X = data.drop(columns=['datetime', 'close', 'symbol'])
        y = data['close']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)

        print(f'{symbol} Model trained. MSE: {mse}')

        # Save model
        model_filename = os.path.join(self.model_dir, f'{symbol}_model.pkl')
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)

    def load_model(self, symbol: str):
        """
        Load the machine learning model for a specific cryptocurrency.

        Args:
            symbol: Cryptocurrency symbol for which to load the model

        Returns:
            Loaded model
        """
        model_filename = os.path.join(self.model_dir, f'{symbol}_model.pkl')
        if not os.path.exists(model_filename):
            raise FileNotFoundError(f'Model for {symbol} not found.')

        with open(model_filename, 'rb') as f:
            model = pickle.load(f)
        return model

    def predict(self, model, data: pd.DataFrame) -> np.ndarray:
        """
        Predict using the loaded model.

        Args:
            model: Pre-trained model
            data: DataFrame containing features

        Returns:
            Predicted values
        """
        prepared_data = self._prepare_features(data)
        X = prepared_data.drop(columns=['datetime', 'close', 'symbol'])
        predictions = model.predict(X)
        return predictions

