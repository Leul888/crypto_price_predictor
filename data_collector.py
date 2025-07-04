"""
Cryptocurrency Data Collector

This module handles fetching and processing cryptocurrency price data
from various sources including Yahoo Finance and cryptocurrency APIs.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CryptoDataCollector:
    """Collects and processes cryptocurrency price data."""
    
    def __init__(self):
        self.supported_cryptos = {
            'BTC': 'BTC-USD',
            'ETH': 'ETH-USD',
            'ADA': 'ADA-USD',
            'DOT': 'DOT-USD',
            'SOL': 'SOL-USD',
            'MATIC': 'MATIC-USD',
            'AVAX': 'AVAX-USD',
            'LINK': 'LINK-USD'
        }
    
    def fetch_crypto_data(self, 
                         symbol: str, 
                         period: str = '1y',
                         interval: str = '1h') -> pd.DataFrame:
        """
        Fetch cryptocurrency data from Yahoo Finance.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            if symbol not in self.supported_cryptos:
                raise ValueError(f"Unsupported cryptocurrency: {symbol}")
            
            ticker_symbol = self.supported_cryptos[symbol]
            ticker = yf.Ticker(ticker_symbol)
            
            logger.info(f"Fetching {symbol} data for period {period} with interval {interval}")
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for {symbol}")
            
            # Clean column names
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            
            # Add symbol column
            data['symbol'] = symbol
            
            # Reset index to make datetime a column
            data.reset_index(inplace=True)
            
            logger.info(f"Successfully fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise
    
    def fetch_multiple_cryptos(self, 
                              symbols: List[str], 
                              period: str = '1y',
                              interval: str = '1h') -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple cryptocurrencies.
        
        Args:
            symbols: List of cryptocurrency symbols
            period: Time period
            interval: Data interval
        
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        data_dict = {}
        
        for symbol in symbols:
            try:
                data = self.fetch_crypto_data(symbol, period, interval)
                data_dict[symbol] = data
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {str(e)}")
                continue
        
        return data_dict
    
    def get_market_data(self, symbols: List[str]) -> pd.DataFrame:
        """
        Get current market data for specified cryptocurrencies.
        
        Args:
            symbols: List of cryptocurrency symbols
        
        Returns:
            DataFrame with current market data
        """
        market_data = []
        
        for symbol in symbols:
            try:
                if symbol not in self.supported_cryptos:
                    continue
                
                ticker_symbol = self.supported_cryptos[symbol]
                ticker = yf.Ticker(ticker_symbol)
                info = ticker.info
                
                market_data.append({
                    'symbol': symbol,
                    'current_price': info.get('regularMarketPrice', 0),
                    'market_cap': info.get('marketCap', 0),
                    'volume_24h': info.get('volume24Hr', 0),
                    'price_change_24h': info.get('regularMarketChangePercent', 0),
                    'timestamp': datetime.now()
                })
                
            except Exception as e:
                logger.error(f"Error fetching market data for {symbol}: {str(e)}")
                continue
        
        return pd.DataFrame(market_data)
    
    def save_data(self, data: pd.DataFrame, filename: str) -> None:
        """
        Save data to CSV file.
        
        Args:
            data: DataFrame to save
            filename: Output filename
        """
        try:
            data.to_csv(filename, index=False)
            logger.info(f"Data saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving data to {filename}: {str(e)}")
            raise
    
    def load_data(self, filename: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            filename: Input filename
        
        Returns:
            DataFrame with loaded data
        """
        try:
            data = pd.read_csv(filename)
            logger.info(f"Data loaded from {filename}")
            return data
        except Exception as e:
            logger.error(f"Error loading data from {filename}: {str(e)}")
            raise


def main():
    """Example usage of the CryptoDataCollector."""
    collector = CryptoDataCollector()
    
    # Fetch Bitcoin data
    btc_data = collector.fetch_crypto_data('BTC', period='6mo', interval='1h')
    print(f"Fetched {len(btc_data)} records for BTC")
    print(btc_data.head())
    
    # Fetch multiple cryptocurrencies
    symbols = ['BTC', 'ETH', 'ADA']
    multi_data = collector.fetch_multiple_cryptos(symbols, period='1mo', interval='1h')
    
    for symbol, data in multi_data.items():
        print(f"{symbol}: {len(data)} records")
    
    # Save Bitcoin data
    collector.save_data(btc_data, 'btc_data.csv')


if __name__ == "__main__":
    main()
