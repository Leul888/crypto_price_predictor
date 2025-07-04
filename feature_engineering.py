"""
Feature Engineering for Cryptocurrency Price Prediction

This module contains functions for creating technical indicators and features
that can be used in machine learning models for cryptocurrency price prediction.
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates technical indicators and features for cryptocurrency prediction."""
    
    def __init__(self):
        self.feature_columns = []
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators added
        """
        df = df.copy()
        
        # Moving Averages
        df['sma_5'] = ta.trend.sma_indicator(df['close'], window=5)
        df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        
        # Exponential Moving Averages
        df['ema_5'] = ta.trend.ema_indicator(df['close'], window=5)
        df['ema_10'] = ta.trend.ema_indicator(df['close'], window=10)
        df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
        
        # Relative Strength Index
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        
        # MACD
        df['macd'] = ta.trend.macd(df['close'])
        df['macd_signal'] = ta.trend.macd_signal(df['close'])
        df['macd_histogram'] = ta.trend.macd_diff(df['close'])
        
        # Bollinger Bands
        df['bb_upper'] = ta.volatility.bollinger_hband(df['close'])
        df['bb_middle'] = ta.volatility.bollinger_mavg(df['close'])
        df['bb_lower'] = ta.volatility.bollinger_lband(df['close'])
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / df['bb_width']
        
        # Stochastic Oscillator
        df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
        df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'])
        
        # Williams %R
        df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
        
        # Average True Range
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        
        # Commodity Channel Index
        df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'])
        
        # Money Flow Index
        df['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'])
        
        # On-Balance Volume
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        
        # Volume indicators
        try:
            df['volume_sma'] = ta.volume.VolumeSMAIndicator(close=df['close'], volume=df['volume']).volume_sma()
        except:
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
        
        try:
            df['volume_weighted_average_price'] = ta.volume.VolumePriceTrendIndicator(
                high=df['high'], low=df['low'], close=df['close'], volume=df['volume']
            ).volume_price_trend()
        except:
            # Calculate VWAP manually if ta function fails
            df['volume_weighted_average_price'] = (
                (df['high'] + df['low'] + df['close']) / 3 * df['volume']
            ).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
        
        logger.info("Technical indicators added successfully")
        return df
    
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price-based features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with price features added
        """
        df = df.copy()
        
        # Price changes
        df['price_change'] = df['close'].pct_change()
        df['price_change_1h'] = df['close'].pct_change(periods=1)
        df['price_change_4h'] = df['close'].pct_change(periods=4)
        df['price_change_24h'] = df['close'].pct_change(periods=24)
        
        # High-Low spread
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        
        # Open-Close spread
        df['oc_spread'] = (df['close'] - df['open']) / df['open']
        
        # Volume change
        df['volume_change'] = df['volume'].pct_change()
        
        # Price volatility (rolling standard deviation)
        df['volatility_5'] = df['close'].rolling(window=5).std()
        df['volatility_10'] = df['close'].rolling(window=10).std()
        df['volatility_20'] = df['close'].rolling(window=20).std()
        
        # Price momentum
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        logger.info("Price features added successfully")
        return df
    
    def create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create basic features for smaller datasets.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with basic features added
        """
        df = df.copy()
        
        try:
            # Basic price features
            df['price_change'] = df['close'].pct_change()
            df['hl_spread'] = (df['high'] - df['low']) / df['close']
            df['oc_spread'] = (df['close'] - df['open']) / df['open']
            
            # Simple moving averages (smaller windows)
            if len(df) >= 5:
                df['sma_5'] = df['close'].rolling(window=5).mean()
            if len(df) >= 10:
                df['sma_10'] = df['close'].rolling(window=10).mean()
            
            # Basic volatility
            if len(df) >= 5:
                df['volatility_5'] = df['close'].rolling(window=5).std()
            
            # Volume features
            df['volume_change'] = df['volume'].pct_change()
            
            # Basic time features
            if 'Datetime' in df.columns:
                df['datetime'] = df['Datetime']
            
            if 'datetime' in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
                    df['datetime'] = pd.to_datetime(df['datetime'])
                df['hour'] = df['datetime'].dt.hour
                df['day_of_week'] = df['datetime'].dt.dayofweek
            
            # Simple lag features
            df['close_lag_1'] = df['close'].shift(1)
            if len(df) >= 3:
                df['close_lag_2'] = df['close'].shift(2)
            
            # Create target (next period price)
            df['target'] = df['close'].shift(-1)
            
            # Drop rows with NaN values
            df = df.dropna()
            
            logger.info(f"Basic features created successfully. Shape: {df.shape}")
            
        except Exception as e:
            logger.error(f"Error creating basic features: {str(e)}")
            raise
        
        return df
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features.
        
        Args:
            df: DataFrame with datetime index or column
            
        Returns:
            DataFrame with time features added
        """
        df = df.copy()
        
        # Ensure datetime column exists
        if 'datetime' not in df.columns:
            if df.index.name == 'Datetime' or pd.api.types.is_datetime64_any_dtype(df.index):
                df['datetime'] = df.index
            elif 'Datetime' in df.columns:
                df['datetime'] = df['Datetime']
            else:
                # Print available columns for debugging
                print(f"Available columns: {df.columns.tolist()}")
                print(f"Index name: {df.index.name}")
                print(f"Index type: {type(df.index)}")
                raise ValueError("No datetime column or index found")
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
            df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Extract time components
        df['hour'] = df['datetime'].dt.hour
        df['day'] = df['datetime'].dt.day
        df['month'] = df['datetime'].dt.month
        df['year'] = df['datetime'].dt.year
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['day_of_year'] = df['datetime'].dt.dayofyear
        df['week_of_year'] = df['datetime'].dt.isocalendar().week
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Market session indicators (assuming UTC time)
        df['is_asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 9)).astype(int)
        df['is_european_session'] = ((df['hour'] >= 8) & (df['hour'] < 17)).astype(int)
        df['is_american_session'] = ((df['hour'] >= 14) & (df['hour'] < 23)).astype(int)
        
        # Weekend indicator
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        logger.info("Time features added successfully")
        return df
    
    def add_lag_features(self, df: pd.DataFrame, 
                        columns: List[str] = ['close', 'volume'], 
                        lags: List[int] = [1, 2, 3, 6, 12, 24]) -> pd.DataFrame:
        """
        Add lag features for specified columns.
        
        Args:
            df: DataFrame with data
            columns: List of column names to create lags for
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features added
        """
        df = df.copy()
        
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        logger.info(f"Lag features added for columns: {columns}")
        return df
    
    def add_rolling_features(self, df: pd.DataFrame, 
                           windows: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """
        Add rolling statistical features.
        
        Args:
            df: DataFrame with data
            windows: List of window sizes for rolling statistics
            
        Returns:
            DataFrame with rolling features added
        """
        df = df.copy()
        
        for window in windows:
            # Rolling statistics for close price
            df[f'close_mean_{window}'] = df['close'].rolling(window=window).mean()
            df[f'close_std_{window}'] = df['close'].rolling(window=window).std()
            df[f'close_min_{window}'] = df['close'].rolling(window=window).min()
            df[f'close_max_{window}'] = df['close'].rolling(window=window).max()
            
            # Rolling statistics for volume
            df[f'volume_mean_{window}'] = df['volume'].rolling(window=window).mean()
            df[f'volume_std_{window}'] = df['volume'].rolling(window=window).std()
            
            # Position within rolling window
            df[f'close_position_{window}'] = (df['close'] - df[f'close_min_{window}']) / (
                df[f'close_max_{window}'] - df[f'close_min_{window}']
            )
        
        logger.info(f"Rolling features added for windows: {windows}")
        return df
    
    def create_target_variable(self, df: pd.DataFrame, 
                             target_col: str = 'close',
                             prediction_horizon: int = 1,
                             target_type: str = 'price') -> pd.DataFrame:
        """
        Create target variable for prediction.
        
        Args:
            df: DataFrame with data
            target_col: Column to use as base for target
            prediction_horizon: Number of periods ahead to predict
            target_type: Type of target ('price', 'return', 'direction')
            
        Returns:
            DataFrame with target variable added
        """
        df = df.copy()
        
        if target_type == 'price':
            df['target'] = df[target_col].shift(-prediction_horizon)
        elif target_type == 'return':
            df['target'] = df[target_col].pct_change(periods=prediction_horizon).shift(-prediction_horizon)
        elif target_type == 'direction':
            future_price = df[target_col].shift(-prediction_horizon)
            df['target'] = (future_price > df[target_col]).astype(int)
        else:
            raise ValueError(f"Unknown target_type: {target_type}")
        
        logger.info(f"Target variable created: {target_type} with horizon {prediction_horizon}")
        return df
    
    def create_all_features(self, df: pd.DataFrame, 
                          target_type: str = 'price',
                          prediction_horizon: int = 1) -> pd.DataFrame:
        """
        Create all features for the dataset.
        
        Args:
            df: DataFrame with OHLCV data
            target_type: Type of target variable to create
            prediction_horizon: Number of periods ahead to predict
            
        Returns:
            DataFrame with all features added
        """
        logger.info("Starting feature engineering...")
        
        # Add all feature types
        df = self.add_technical_indicators(df)
        df = self.add_price_features(df)
        df = self.add_time_features(df)
        df = self.add_lag_features(df)
        df = self.add_rolling_features(df)
        df = self.create_target_variable(df, target_type=target_type, 
                                       prediction_horizon=prediction_horizon)
        
        # Remove rows with NaN values
        initial_rows = len(df)
        df = df.dropna()
        final_rows = len(df)
        
        logger.info(f"Feature engineering completed. Rows: {initial_rows} -> {final_rows}")
        
        return df
    
    def get_feature_importance(self, model, feature_names: List[str]) -> pd.DataFrame:
        """
        Get feature importance from a trained model.
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance scores
        """
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            raise ValueError("Model does not have feature_importances_ attribute")


def main():
    """Example usage of FeatureEngineer."""
    # This would typically use real data from data_collector
    # For demonstration, create sample data
    dates = pd.date_range('2023-01-01', periods=1000, freq='H')
    
    # Generate sample OHLCV data
    np.random.seed(42)
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
    featured_data = fe.create_all_features(sample_data)
    
    print(f"Original columns: {len(sample_data.columns)}")
    print(f"Featured columns: {len(featured_data.columns)}")
    print(f"Featured data shape: {featured_data.shape}")
    print("\nSample features:")
    print(featured_data.head())


if __name__ == "__main__":
    main()
