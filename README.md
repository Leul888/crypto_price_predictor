# Cryptocurrency Price Predictor

A comprehensive machine learning system for cryptocurrency price prediction with an interactive web interface built using Streamlit.

## 🚀 Features

- **Real-time Data Collection**: Fetch historical cryptocurrency data from Yahoo Finance
- **Advanced Feature Engineering**: 30+ technical indicators including RSI, MACD, Bollinger Bands, and more
- **Multiple ML Models**: Compare 7 different algorithms (Random Forest, Gradient Boosting, Neural Networks, etc.)
- **Interactive Web Interface**: Beautiful Streamlit dashboard with real-time visualizations
- **Model Performance Analysis**: Comprehensive evaluation with RMSE, MAE, and R² metrics
- **Hyperparameter Tuning**: Automated optimization using GridSearchCV

## 🔧 Technologies Used

- **Python 3.8+**
- **Machine Learning**: Scikit-learn, NumPy, Pandas
- **Technical Analysis**: TA-Lib
- **Data Source**: Yahoo Finance (yfinance)
- **Web Interface**: Streamlit
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Model Persistence**: Joblib

## 📈 Supported Cryptocurrencies

- Bitcoin (BTC)
- Ethereum (ETH)
- Cardano (ADA)
- Polkadot (DOT)
- Solana (SOL)
- Polygon (MATIC)
- Avalanche (AVAX)
- Chainlink (LINK)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/crypto_price_predictor.git
cd crypto_price_predictor
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Command Line Interface
Run the main application:
```bash
python main.py
```

### Web Interface
Launch the Streamlit app:
```bash
streamlit run streamlit_app.py
```

### Individual Components

**Data Collection**:
```python
from data_collector import CryptoDataCollector

collector = CryptoDataCollector()
btc_data = collector.fetch_crypto_data('BTC', period='6mo', interval='1h')
```

**Feature Engineering**:
```python
from feature_engineering import FeatureEngineer

engineer = FeatureEngineer()
featured_data = engineer.create_all_features(btc_data, target_type='price', prediction_horizon=1)
```

**Model Training**:
```python
from ml_models import CryptoPricePredictor

predictor = CryptoPricePredictor()
results = predictor.train_all_models(featured_data)
```

## 📊 Machine Learning Models

The system implements and compares the following algorithms:

1. **Random Forest Regressor**
2. **Gradient Boosting Regressor**
3. **Linear Regression**
4. **Ridge Regression**
5. **Lasso Regression**
6. **Support Vector Regression (SVR)**
7. **Multi-layer Perceptron (Neural Network)**

## 📈 Technical Indicators

The feature engineering module creates 30+ technical indicators:

- **Trend Indicators**: SMA, EMA, MACD
- **Momentum Indicators**: RSI, Stochastic Oscillator, Williams %R
- **Volatility Indicators**: Bollinger Bands, ATR
- **Volume Indicators**: OBV, MFI, Volume SMA
- **Custom Features**: Price spreads, volatility measures, momentum features

## 🎯 Model Evaluation

Models are evaluated using multiple metrics:
- **RMSE** (Root Mean Square Error)
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)
- **Cross-validation scores**

## 📁 Project Structure

```
crypto_price_predictor/
├── data_collector.py          # Data fetching and processing
├── feature_engineering.py     # Technical indicators and features
├── ml_models.py               # Machine learning models and evaluation
├── visualization.py           # Data visualization utilities
├── streamlit_app.py           # Main Streamlit web interface
├── simple_streamlit_app.py    # Simplified web interface
├── price_predictor.py         # Core prediction logic
├── main.py                    # Command-line interface
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## 🔮 Future Enhancements

- [ ] Real-time prediction updates
- [ ] Additional cryptocurrency exchanges
- [ ] Advanced deep learning models (LSTM, GRU)
- [ ] Portfolio optimization features
- [ ] Risk management indicators
- [ ] Mobile-responsive design
- [ ] API endpoints for predictions

## ⚠️ Disclaimer

This project is for educational and research purposes only. Cryptocurrency trading involves significant financial risk. The predictions made by this system should not be considered as financial advice. Always do your own research and consult with financial professionals before making investment decisions.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

Feel free to reach out if you have any questions or suggestions!

---

**Built with ❤️ for the crypto community**
