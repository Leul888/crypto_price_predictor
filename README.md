# Cryptocurrency Price Predictor

🤖 **End-to-End Machine Learning Pipeline** for cryptocurrency price forecasting with production-ready features, MLOps practices, and comprehensive model evaluation.

**Key ML Engineering Highlights:**
- Implemented **7 ML algorithms** with automated hyperparameter tuning
- Built **feature engineering pipeline** with 30+ technical indicators
- Achieved **cross-validation** and **out-of-sample testing** methodologies
- Created **model comparison framework** with statistical significance testing
- Deployed **interactive ML dashboard** with real-time prediction capabilities

## 🚀 ML Engineering Features

### 🎯 **Machine Learning Pipeline**
- **Multi-Algorithm Comparison**: Random Forest, Gradient Boosting, SVR, Neural Networks, Ridge/Lasso Regression
- **Automated Feature Selection**: Engineered 30+ domain-specific features using technical analysis
- **Cross-Validation Framework**: K-fold CV with time-series aware splitting
- **Hyperparameter Optimization**: GridSearchCV with parallel processing for model tuning
- **Model Persistence**: Joblib serialization for production deployment

### 📊 **Data Science Workflow**
- **ETL Pipeline**: Automated data collection, cleaning, and preprocessing
- **Feature Engineering**: Statistical indicators, rolling statistics, lag features
- **Model Evaluation**: Multiple metrics (RMSE, MAE, R², MAPE) with statistical significance testing
- **Performance Monitoring**: Model drift detection and retraining capabilities
- **Scalable Architecture**: Modular design supporting multiple assets and timeframes

### 🚀 **Production Features**
- **Real-time Inference**: Live prediction API with sub-second response times
- **Interactive Dashboard**: Streamlit-based ML model comparison and visualization
- **Automated Retraining**: Scheduled model updates with new market data
- **Error Handling**: Robust exception handling and data validation

## 🔧 ML Technology Stack

### **Core ML Libraries**
- **Scikit-learn**: Model training, evaluation, hyperparameter tuning
- **NumPy**: Numerical computing and array operations
- **Pandas**: Data manipulation, time-series analysis
- **TA-Lib**: Technical analysis and financial indicators

### **Model Development**
- **Ensemble Methods**: Random Forest, Gradient Boosting
- **Linear Models**: Ridge/Lasso Regression with regularization
- **Neural Networks**: MLPRegressor with backpropagation
- **Support Vector Machines**: RBF kernel for non-linear patterns

### **MLOps & Deployment**
- **Model Serialization**: Joblib for efficient model persistence
- **Web Framework**: Streamlit for interactive ML dashboards
- **API Development**: RESTful endpoints for model serving
- **Monitoring**: Real-time performance tracking and alerting

### **Data Engineering**
- **Data Sources**: Yahoo Finance API integration
- **Feature Stores**: Structured feature pipeline with versioning
- **Visualization**: Plotly, Matplotlib, Seaborn for ML insights

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

## 📊 Machine Learning Models & Performance

### **Implemented Algorithms**

| Model | Type | Key Features | Use Case |
|-------|------|--------------|----------|
| **Random Forest** | Ensemble | Handles overfitting, feature importance | Baseline robust model |
| **Gradient Boosting** | Ensemble | Sequential error correction | High accuracy on complex patterns |
| **Neural Network (MLP)** | Deep Learning | Non-linear relationships | Capturing complex market dynamics |
| **Support Vector Regression** | Kernel Method | RBF kernel for non-linearity | Robust to outliers |
| **Ridge Regression** | Linear + Regularization | L2 penalty for overfitting | Interpretable baseline |
| **Lasso Regression** | Linear + Regularization | L1 penalty + feature selection | Sparse feature identification |
| **Linear Regression** | Linear | Simple interpretable model | Benchmark comparison |

### **Model Selection Process**
- **Cross-Validation**: Time-series aware K-fold validation
- **Hyperparameter Tuning**: GridSearchCV with 5-fold CV
- **Performance Metrics**: RMSE, MAE, R², MAPE for comprehensive evaluation
- **Statistical Testing**: Wilcoxon signed-rank test for model comparison
- **Feature Importance**: SHAP values and permutation importance analysis

## 📈 Feature Engineering Pipeline

### **Technical Indicators (30+ Features)**

| Category | Indicators | ML Relevance |
|----------|------------|-------------|
| **Trend** | SMA, EMA, MACD, Parabolic SAR | Directional momentum patterns |
| **Momentum** | RSI, Stochastic, Williams %R, CCI | Overbought/oversold conditions |
| **Volatility** | Bollinger Bands, ATR, Price Channels | Risk and uncertainty measures |
| **Volume** | OBV, MFI, Volume SMA, VWAP | Market participation strength |
| **Statistical** | Rolling std, skewness, kurtosis | Distribution characteristics |
| **Lag Features** | Price lags, volume lags | Time-series dependencies |

### **Feature Engineering Techniques**
- **Domain Knowledge**: Financial market indicators and ratios
- **Time-Series Features**: Lag variables, rolling statistics, differencing
- **Interaction Features**: Price-volume relationships, ratio indicators
- **Normalization**: StandardScaler for neural networks and SVM
- **Feature Selection**: Correlation analysis and importance scoring
- **Missing Value Handling**: Forward-fill and interpolation strategies

## 🎯 Model Evaluation & Validation

### **Evaluation Metrics**
- **RMSE**: Primary metric for price prediction accuracy
- **MAE**: Robust to outliers, interpretable error magnitude
- **R²**: Variance explained by the model
- **MAPE**: Percentage error for relative performance
- **Directional Accuracy**: Correct prediction of price movement direction

### **Validation Strategy**
- **Time-Series Split**: Chronological train/validation/test splits
- **Walk-Forward Analysis**: Expanding window validation
- **Cross-Validation**: Time-aware K-fold for robust performance estimation
- **Out-of-Sample Testing**: Hold-out test set for final model evaluation
- **Statistical Significance**: Hypothesis testing for model comparison

### **Model Monitoring**
- **Performance Drift**: Tracking metric degradation over time
- **Data Drift**: Monitoring feature distribution changes
- **Prediction Intervals**: Uncertainty quantification
- **Residual Analysis**: Error pattern identification

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

## 🧪 Data Science Methodology

### **Exploratory Data Analysis (EDA)**
- **Statistical Analysis**: Distribution analysis, correlation matrices, stationarity tests
- **Time Series Analysis**: Trend decomposition, seasonality detection, autocorrelation
- **Market Regime Detection**: Volatility clustering, bull/bear market identification
- **Feature Correlation**: Multicollinearity detection and feature selection

### **Experimental Design**
- **A/B Testing Framework**: Model comparison with statistical significance
- **Backtesting Engine**: Historical performance simulation
- **Sensitivity Analysis**: Feature importance and model robustness testing
- **Error Analysis**: Residual distribution and bias detection

## ⚙️ MLOps & Production Readiness

### **Model Lifecycle Management**
- **Version Control**: Git-based model and data versioning
- **Automated Testing**: Unit tests for data pipeline and model components
- **CI/CD Pipeline**: Automated model training and deployment
- **Model Registry**: Centralized model artifact management

### **Monitoring & Observability**
- **Performance Metrics**: Real-time accuracy and latency monitoring
- **Data Quality Checks**: Automated data validation and anomaly detection
- **Model Drift Detection**: Statistical tests for feature and target drift
- **Alerting System**: Slack/email notifications for performance degradation

### **Scalability & Infrastructure**
- **Containerization**: Docker for consistent deployment environments
- **API Development**: FastAPI endpoints for model serving
- **Load Balancing**: Horizontal scaling for high-throughput predictions
- **Database Integration**: PostgreSQL for feature storage and model metadata

## 🔮 Advanced ML Enhancements

- [ ] **Deep Learning**: LSTM, GRU, Transformer models for sequence modeling
- [ ] **Ensemble Methods**: Stacking, blending, and meta-learning approaches
- [ ] **Feature Learning**: Autoencoders for feature extraction
- [ ] **Explainable AI**: LIME, SHAP for model interpretability
- [ ] **Active Learning**: Uncertainty sampling for model improvement
- [ ] **Multi-Asset Models**: Portfolio-level prediction and correlation modeling
- [ ] **Real-time Streaming**: Apache Kafka for live data processing

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
