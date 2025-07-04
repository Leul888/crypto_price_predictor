"""
Visualization Module for Cryptocurrency Price Prediction

This module provides plotting functions for data analysis,
model performance, and prediction results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class CryptoVisualizer:
    """Visualization utilities for cryptocurrency prediction analysis."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def plot_price_history(self, data: pd.DataFrame, symbol: str = 'BTC', 
                          show_volume: bool = True) -> None:
        """
        Plot cryptocurrency price history with volume.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Cryptocurrency symbol
            show_volume: Whether to show volume subplot
        """
        if show_volume:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, 
                                          gridspec_kw={'height_ratios': [3, 1]})
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=self.figsize)
        
        # Price plot
        ax1.plot(data['datetime'], data['close'], label='Close Price', linewidth=1.5)
        ax1.set_title(f'{symbol} Price History', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price (USD)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Volume plot
        if show_volume:
            ax2.bar(data['datetime'], data['volume'], alpha=0.7, color='gray')
            ax2.set_title('Trading Volume', fontsize=14)
            ax2.set_ylabel('Volume', fontsize=12)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_interactive_price_chart(self, data: pd.DataFrame, symbol: str = 'BTC') -> go.Figure:
        """
        Create interactive price chart with technical indicators.
        
        Args:
            data: DataFrame with OHLCV and indicator data
            symbol: Cryptocurrency symbol
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=[f'{symbol} Price', 'Volume', 'RSI'],
            vertical_spacing=0.08,
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data['datetime'],
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add moving averages if available
        if 'sma_20' in data.columns:
            fig.add_trace(
                go.Scatter(x=data['datetime'], y=data['sma_20'], 
                          name='SMA 20', line=dict(color='orange')),
                row=1, col=1
            )
        
        if 'sma_50' in data.columns:
            fig.add_trace(
                go.Scatter(x=data['datetime'], y=data['sma_50'], 
                          name='SMA 50', line=dict(color='red')),
                row=1, col=1
            )
        
        # Volume
        fig.add_trace(
            go.Bar(x=data['datetime'], y=data['volume'], name='Volume'),
            row=2, col=1
        )
        
        # RSI if available
        if 'rsi' in data.columns:
            fig.add_trace(
                go.Scatter(x=data['datetime'], y=data['rsi'], 
                          name='RSI', line=dict(color='purple')),
                row=3, col=1
            )
            
            # RSI overbought/oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        fig.update_layout(
            title=f'{symbol} Technical Analysis',
            xaxis_rangeslider_visible=False,
            height=800
        )
        
        return fig
    
    def plot_model_comparison(self, results_df: pd.DataFrame) -> None:
        """
        Plot model performance comparison.
        
        Args:
            results_df: DataFrame with model performance metrics
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # RMSE comparison
        sns.barplot(data=results_df, x='model_name', y='test_rmse', ax=ax1)
        ax1.set_title('Test RMSE by Model', fontsize=14, fontweight='bold')
        ax1.set_ylabel('RMSE')
        ax1.tick_params(axis='x', rotation=45)
        
        # R² comparison
        sns.barplot(data=results_df, x='model_name', y='test_r2', ax=ax2)
        ax2.set_title('Test R² by Model', fontsize=14, fontweight='bold')
        ax2.set_ylabel('R² Score')
        ax2.tick_params(axis='x', rotation=45)
        
        # MAE comparison
        sns.barplot(data=results_df, x='model_name', y='test_mae', ax=ax3)
        ax3.set_title('Test MAE by Model', fontsize=14, fontweight='bold')
        ax3.set_ylabel('MAE')
        ax3.tick_params(axis='x', rotation=45)
        
        # Training vs Test RMSE
        train_test_data = pd.melt(
            results_df[['model_name', 'train_rmse', 'test_rmse']], 
            id_vars=['model_name'], 
            value_vars=['train_rmse', 'test_rmse'],
            var_name='dataset', value_name='rmse'
        )
        sns.barplot(data=train_test_data, x='model_name', y='rmse', 
                   hue='dataset', ax=ax4)
        ax4.set_title('Train vs Test RMSE', fontsize=14, fontweight='bold')
        ax4.set_ylabel('RMSE')
        ax4.tick_params(axis='x', rotation=45)
        ax4.legend(title='Dataset')
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, top_n: int = 20) -> None:
        """
        Plot feature importance.
        
        Args:
            importance_df: DataFrame with feature importance scores
            top_n: Number of top features to show
        """
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=self.figsize)
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importance', fontsize=16, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()
        plt.show()
    
    def plot_predictions_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  model_name: str = 'Model') -> None:
        """
        Plot predictions vs actual values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot
        ax1.scatter(y_true, y_pred, alpha=0.6, color=self.colors[0])
        ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                'r--', lw=2, label='Perfect Prediction')
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title(f'{model_name}: Predictions vs Actual')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Residuals plot
        residuals = y_true - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.6, color=self.colors[1])
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Residuals')
        ax2.set_title(f'{model_name}: Residuals Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_prediction_timeline(self, data: pd.DataFrame, predictions: np.ndarray,
                               actual_col: str = 'close', model_name: str = 'Model') -> None:
        """
        Plot prediction timeline against actual values.
        
        Args:
            data: DataFrame with datetime and actual values
            predictions: Predicted values
            actual_col: Column name for actual values
            model_name: Name of the model
        """
        plt.figure(figsize=self.figsize)
        
        # Ensure we have the same length
        min_len = min(len(data), len(predictions))
        data_subset = data.iloc[:min_len].copy()
        predictions_subset = predictions[:min_len]
        
        plt.plot(data_subset['datetime'], data_subset[actual_col], 
                label='Actual', linewidth=1.5, color=self.colors[0])
        plt.plot(data_subset['datetime'], predictions_subset, 
                label='Predicted', linewidth=1.5, color=self.colors[1])
        
        plt.title(f'{model_name}: Prediction Timeline', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_matrix(self, data: pd.DataFrame, features: Optional[List[str]] = None) -> None:
        """
        Plot correlation matrix of features.
        
        Args:
            data: DataFrame with features
            features: List of features to include (if None, use all numeric columns)
        """
        if features is None:
            # Select only numeric columns
            numeric_data = data.select_dtypes(include=[np.number])
        else:
            numeric_data = data[features]
        
        # Calculate correlation matrix
        corr_matrix = numeric_data.corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5)
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_price_distribution(self, data: pd.DataFrame, price_col: str = 'close') -> None:
        """
        Plot price distribution and statistics.
        
        Args:
            data: DataFrame with price data
            price_col: Column name for price data
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Histogram
        ax1.hist(data[price_col], bins=50, alpha=0.7, color=self.colors[0])
        ax1.set_title('Price Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Price')
        ax1.set_ylabel('Frequency')
        
        # Box plot
        ax2.boxplot(data[price_col])
        ax2.set_title('Price Box Plot', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Price')
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(data[price_col], dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot', fontsize=14, fontweight='bold')
        
        # Price over time
        ax4.plot(data['datetime'], data[price_col], linewidth=1)
        ax4.set_title('Price Over Time', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Price')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()


def create_dashboard_plots(data: pd.DataFrame, results_df: pd.DataFrame, 
                          predictions: np.ndarray, symbol: str = 'BTC') -> None:
    """
    Create a comprehensive dashboard of plots.
    
    Args:
        data: DataFrame with cryptocurrency data
        results_df: DataFrame with model results
        predictions: Model predictions
        symbol: Cryptocurrency symbol
    """
    visualizer = CryptoVisualizer()
    
    print(f"Creating comprehensive dashboard for {symbol}...")
    
    # 1. Price history
    print("1. Price History Plot")
    visualizer.plot_price_history(data, symbol)
    
    # 2. Model comparison
    print("2. Model Performance Comparison")
    visualizer.plot_model_comparison(results_df)
    
    # 3. Predictions vs actual
    print("3. Predictions vs Actual")
    y_true = data['target'].dropna().values
    min_len = min(len(y_true), len(predictions))
    visualizer.plot_predictions_vs_actual(y_true[:min_len], predictions[:min_len])
    
    # 4. Price distribution
    print("4. Price Distribution Analysis")
    visualizer.plot_price_distribution(data)
    
    print("Dashboard creation complete!")


def main():
    """Example usage of visualization functions."""
    # This would typically use real data from the main pipeline
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
        'target': close_prices + np.random.randn(1000) * 0.1
    })
    
    # Sample model results
    results_data = {
        'model_name': ['Random Forest', 'Linear Regression', 'SVR'],
        'test_rmse': [0.5, 0.8, 0.6],
        'test_r2': [0.85, 0.70, 0.80],
        'test_mae': [0.3, 0.6, 0.4],
        'train_rmse': [0.3, 0.7, 0.5],
        'train_r2': [0.95, 0.75, 0.85]
    }
    results_df = pd.DataFrame(results_data)
    
    # Create visualizations
    visualizer = CryptoVisualizer()
    
    print("Creating sample visualizations...")
    visualizer.plot_price_history(sample_data)
    visualizer.plot_model_comparison(results_df)
    
    # Sample predictions
    predictions = sample_data['close'].values + np.random.randn(1000) * 0.2
    visualizer.plot_predictions_vs_actual(sample_data['close'].values, predictions)


if __name__ == "__main__":
    main()
