import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
try:
    from data_collector import CryptoDataCollector
    from feature_engineering import FeatureEngineer
    from ml_models import CryptoPricePredictor
    from visualization import CryptoVisualizer
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Configure Streamlit page
st.set_page_config(
    page_title="Crypto Price Predictor",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">₿ Cryptocurrency Price Predictor</h1>', unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.model_trained = False

# Sidebar configuration
st.sidebar.markdown("## 🔧 Configuration")

# Cryptocurrency selection
supported_cryptos = ['BTC', 'ETH', 'ADA', 'DOT', 'SOL', 'MATIC', 'AVAX', 'LINK']
selected_symbol = st.sidebar.selectbox(
    "Select Cryptocurrency",
    supported_cryptos,
    help="Choose the cryptocurrency you want to analyze"
)

# Time period selection
period_options = {
    '1 Day': '1d',
    '5 Days': '5d', 
    '1 Month': '1mo',
    '3 Months': '3mo',
    '6 Months': '6mo',
    '1 Year': '1y',
    '2 Years': '2y'
}
selected_period = st.sidebar.selectbox(
    "Select Time Period",
    list(period_options.keys()),
    index=4,  # Default to 6 months
    help="Choose the historical data period"
)

# Interval selection
interval_options = {
    '1 Hour': '1h',
    '1 Day': '1d',
    '1 Week': '1wk'
}
selected_interval = st.sidebar.selectbox(
    "Select Data Interval",
    list(interval_options.keys()),
    index=0,  # Default to 1 hour
    help="Choose the data granularity"
)

# Initialize components
@st.cache_resource
def initialize_components():
    collector = CryptoDataCollector()
    engineer = FeatureEngineer()
    predictor = CryptoPricePredictor()
    visualizer = CryptoVisualizer()
    return collector, engineer, predictor, visualizer

collector, engineer, predictor, visualizer = initialize_components()

# Data loading section
st.markdown('<h2 class="sub-header">📊 Data Loading</h2>', unsafe_allow_html=True)

if st.button("Load Data", type="primary"):
    with st.spinner(f"Loading {selected_symbol} data..."):
        try:
            # Fetch data
            data = collector.fetch_crypto_data(
                symbol=selected_symbol,
                period=period_options[selected_period],
                interval=interval_options[selected_interval]
            )
            
            st.session_state.data = data
            st.session_state.data_loaded = True
            st.success(f"Successfully loaded {len(data)} records for {selected_symbol}")
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.session_state.data_loaded = False

# Display data if loaded
if st.session_state.data_loaded:
    data = st.session_state.data
    
    # Key metrics
    st.markdown('<h2 class="sub-header">📈 Key Metrics</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = data['close'].iloc[-1]
        st.metric("Current Price", f"${current_price:.2f}")
    
    with col2:
        price_change = data['close'].iloc[-1] - data['close'].iloc[-2]
        price_change_pct = (price_change / data['close'].iloc[-2]) * 100
        st.metric("24h Change", f"{price_change_pct:.2f}%", f"${price_change:.2f}")
    
    with col3:
        high_24h = data['high'].tail(24).max()
        st.metric("24h High", f"${high_24h:.2f}")
    
    with col4:
        low_24h = data['low'].tail(24).min()
        st.metric("24h Low", f"${low_24h:.2f}")
    
    # Data preview
    st.markdown('<h2 class="sub-header">📋 Data Preview</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(data.tail(10), use_container_width=True)
    
    with col2:
        st.markdown("**Dataset Info:**")
        st.write(f"- Total Records: {len(data):,}")
        # Check what column contains the date/time information
        date_col = None
        for col in ['Datetime', 'datetime', 'date', 'Date']:
            if col in data.columns:
                date_col = col
                break
        
        if date_col:
            st.write(f"- Date Range: {data[date_col].min().strftime('%Y-%m-%d')} to {data[date_col].max().strftime('%Y-%m-%d')}")
        else:
            st.write("- Date Range: Not available")
        st.write(f"- Interval: {selected_interval}")
        st.write(f"- Symbol: {selected_symbol}")
    
    # Interactive price chart
    st.markdown('<h2 class="sub-header">📈 Price Chart</h2>', unsafe_allow_html=True)
    
    # Create interactive candlestick chart
    fig = go.Figure()
    
    # Find the date column (use the same one found earlier)
    if 'date_col' not in locals():
        date_col = None
        for col in ['Datetime', 'datetime', 'date', 'Date']:
            if col in data.columns:
                date_col = col
                break
    
    if date_col is None:
        st.error("No date column found in data")
        st.stop()
    
    # Add candlestick
    fig.add_trace(go.Candlestick(
        x=data[date_col],
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close'],
        name='Price'
    ))
    
    fig.update_layout(
        title=f'{selected_symbol} Price Chart',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        height=500,
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Volume chart
    st.markdown('<h2 class="sub-header">📊 Volume Chart</h2>', unsafe_allow_html=True)
    
    fig_volume = go.Figure()
    fig_volume.add_trace(go.Bar(
        x=data[date_col],
        y=data['volume'],
        name='Volume',
        marker_color='lightblue'
    ))
    
    fig_volume.update_layout(
        title=f'{selected_symbol} Trading Volume',
        xaxis_title='Date',
        yaxis_title='Volume',
        height=300
    )
    
    st.plotly_chart(fig_volume, use_container_width=True)
    
    # Feature Engineering Section
    st.markdown('<h2 class="sub-header">🔧 Feature Engineering</h2>', unsafe_allow_html=True)
    
    if st.button("Generate Features", type="primary"):
        with st.spinner("Generating features..."):
            try:
                # Create features
                featured_data = engineer.create_all_features(
                    data, 
                    target_type='price', 
                    prediction_horizon=1
                )
                
                st.session_state.featured_data = featured_data
                st.success(f"Generated {len(featured_data.columns)} features")
                
                # Show feature preview
                st.markdown("**Feature Preview:**")
                feature_cols = [col for col in featured_data.columns if col not in ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
                st.dataframe(featured_data[feature_cols].head(), use_container_width=True)
                
            except Exception as e:
                st.error(f"Error generating features: {str(e)}")
    
    # Model Training Section
    if 'featured_data' in st.session_state:
        st.markdown('<h2 class="sub-header">🤖 Model Training</h2>', unsafe_allow_html=True)
        
        if st.button("Train Models", type="primary"):
            with st.spinner("Training models..."):
                try:
                    # Train all models
                    results = predictor.train_all_models(st.session_state.featured_data)
                    
                    st.session_state.model_results = results
                    st.session_state.model_trained = True
                    st.success("Models trained successfully!")
                    
                    # Display results
                    st.markdown("**Model Performance:**")
                    st.dataframe(results[['model_name', 'test_rmse', 'test_r2', 'test_mae']], use_container_width=True)
                    
                    # Plot model comparison
                    fig_comparison = go.Figure()
                    
                    fig_comparison.add_trace(go.Bar(
                        x=results['model_name'],
                        y=results['test_rmse'],
                        name='RMSE',
                        marker_color='lightcoral'
                    ))
                    
                    fig_comparison.update_layout(
                        title='Model Performance Comparison (RMSE)',
                        xaxis_title='Model',
                        yaxis_title='RMSE',
                        height=400
                    )
                    
                    st.plotly_chart(fig_comparison, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error training models: {str(e)}")
    
    # Prediction Section
    if st.session_state.get('model_trained', False):
        st.markdown('<h2 class="sub-header">🔮 Predictions</h2>', unsafe_allow_html=True)
        
        results = st.session_state.model_results
        best_model = results.iloc[0]['model_name']
        
        st.info(f"Best performing model: **{best_model}** (RMSE: {results.iloc[0]['test_rmse']:.4f})")
        
        # Future predictions (simplified)
        st.markdown("**Next Hour Prediction:**")
        
        # Get the latest features for prediction
        latest_features = st.session_state.featured_data.iloc[-1:]
        
        try:
            # Make prediction using the best model
            X, y, feature_names = predictor.prepare_features(st.session_state.featured_data)
            
            # Get the trained model
            best_model_obj = predictor.models[best_model]
            
            # Make prediction on the latest data point
            latest_X = X[-1:]
            prediction = best_model_obj.predict(latest_X)[0]
            
            current_price = data['close'].iloc[-1]
            price_change = prediction - current_price
            price_change_pct = (price_change / current_price) * 100
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Price", f"${current_price:.2f}")
            
            with col2:
                st.metric("Predicted Price", f"${prediction:.2f}")
            
            with col3:
                st.metric("Expected Change", f"{price_change_pct:.2f}%", f"${price_change:.2f}")
        
        except Exception as e:
            st.error(f"Error making predictions: {str(e)}")

else:
    st.info("👆 Click 'Load Data' to start analyzing cryptocurrency data!")
    
    # Show sample data structure
    st.markdown('<h2 class="sub-header">📖 About This App</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    This cryptocurrency price predictor app provides:
    
    1. **Real-time Data Loading**: Fetch historical cryptocurrency data from Yahoo Finance
    2. **Interactive Visualizations**: Candlestick charts, volume analysis, and more
    3. **Feature Engineering**: Advanced technical indicators and market features
    4. **Machine Learning Models**: Multiple ML algorithms for price prediction
    5. **Performance Comparison**: Compare different models and their accuracy
    
    **Supported Cryptocurrencies**: BTC, ETH, ADA, DOT, SOL, MATIC, AVAX, LINK
    
    **How to use:**
    - Select a cryptocurrency from the sidebar
    - Choose your preferred time period and interval
    - Click 'Load Data' to fetch historical data
    - Generate features and train models
    - View predictions and analysis
    """)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>Built with ❤️ using Streamlit | Cryptocurrency Price Predictor</div>", unsafe_allow_html=True)

