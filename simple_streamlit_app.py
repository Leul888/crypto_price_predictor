import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
try:
    from data_collector import CryptoDataCollector
    from feature_engineering import FeatureEngineer
    from ml_models import CryptoPricePredictor
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
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">₿ Cryptocurrency Price Predictor</h1>', unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'features_generated' not in st.session_state:
    st.session_state.features_generated = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

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
    return collector

collector = initialize_components()

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
        if len(data) > 1:
            price_change = data['close'].iloc[-1] - data['close'].iloc[-2]
            price_change_pct = (price_change / data['close'].iloc[-2]) * 100
            st.metric("Last Change", f"{price_change_pct:.2f}%", f"${price_change:.2f}")
        else:
            st.metric("Last Change", "N/A")
    
    with col3:
        high_price = data['high'].max()
        st.metric(f"Highest ({selected_period})", f"${high_price:.2f}")
    
    with col4:
        low_price = data['low'].min()
        st.metric(f"Lowest ({selected_period})", f"${low_price:.2f}")
    
    # Data preview
    st.markdown('<h2 class="sub-header">📋 Data Preview</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(data.tail(10), use_container_width=True)
    
    with col2:
        st.markdown("**Dataset Info:**")
        st.write(f"- Total Records: {len(data):,}")
        
        # Find date column
        date_col = None
        for col in ['Datetime', 'datetime', 'date', 'Date']:
            if col in data.columns:
                date_col = col
                break
        
        if date_col:
            # Convert to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
                data[date_col] = pd.to_datetime(data[date_col])
            
            min_date = data[date_col].min()
            max_date = data[date_col].max()
            
            st.write(f"- Date Range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
        else:
            st.write("- Date Range: Not available")
        
        st.write(f"- Interval: {selected_interval}")
        st.write(f"- Symbol: {selected_symbol}")
        
        # Show all columns
        st.write("**Columns:**")
        for col in data.columns:
            st.write(f"  • {col}")
    
    # Interactive price chart
    st.markdown('<h2 class="sub-header">📈 Price Chart</h2>', unsafe_allow_html=True)
    
    # Create interactive candlestick chart
    fig = go.Figure()
    
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
    
    # Simple statistics
    st.markdown('<h2 class="sub-header">📊 Statistics</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Price Statistics:**")
        st.write(f"- Mean Price: ${data['close'].mean():.2f}")
        st.write(f"- Median Price: ${data['close'].median():.2f}")
        st.write(f"- Standard Deviation: ${data['close'].std():.2f}")
        st.write(f"- Min Price: ${data['close'].min():.2f}")
        st.write(f"- Max Price: ${data['close'].max():.2f}")
    
    with col2:
        st.markdown("**Volume Statistics:**")
        st.write(f"- Mean Volume: {data['volume'].mean():,.0f}")
        st.write(f"- Median Volume: {data['volume'].median():,.0f}")
        st.write(f"- Total Volume: {data['volume'].sum():,.0f}")
        st.write(f"- Min Volume: {data['volume'].min():,.0f}")
        st.write(f"- Max Volume: {data['volume'].max():,.0f}")
    
    # Price trend analysis
    st.markdown('<h2 class="sub-header">📈 Price Trend Analysis</h2>', unsafe_allow_html=True)
    
    # Calculate simple moving averages if we have enough data
    if len(data) >= 20:
        data['SMA_10'] = data['close'].rolling(window=10).mean()
        data['SMA_20'] = data['close'].rolling(window=20).mean()
        
        # Create line chart with moving averages
        fig_trend = go.Figure()
        
        fig_trend.add_trace(go.Scatter(
            x=data[date_col],
            y=data['close'],
            mode='lines',
            name='Close Price',
            line=dict(color='blue')
        ))
        
        fig_trend.add_trace(go.Scatter(
            x=data[date_col],
            y=data['SMA_10'],
            mode='lines',
            name='10-Period SMA',
            line=dict(color='orange')
        ))
        
        fig_trend.add_trace(go.Scatter(
            x=data[date_col],
            y=data['SMA_20'],
            mode='lines',
            name='20-Period SMA',
            line=dict(color='red')
        ))
        
        fig_trend.update_layout(
            title=f'{selected_symbol} Price with Moving Averages',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            height=400
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Trend interpretation
        current_price = data['close'].iloc[-1]
        sma_10 = data['SMA_10'].iloc[-1]
        sma_20 = data['SMA_20'].iloc[-1]
        
        if current_price > sma_10 > sma_20:
            trend = "📈 **Bullish Trend** - Price is above both moving averages"
        elif current_price < sma_10 < sma_20:
            trend = "📉 **Bearish Trend** - Price is below both moving averages"
        else:
            trend = "➡️ **Sideways/Mixed Trend** - Price is between moving averages"
        
        st.info(trend)
    else:
        st.info("Need at least 20 data points for moving average analysis. Try selecting a longer time period.")
    
    # Machine Learning Section
    st.markdown('<h2 class="sub-header">🤖 Machine Learning Models</h2>', unsafe_allow_html=True)
    
    # ML Model Configuration in Sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🤖 ML Configuration")
    
    # Available models
    available_models = {
        'Random Forest': 'random_forest',
        'Gradient Boosting': 'gradient_boosting',
        'Linear Regression': 'linear_regression',
        'Ridge Regression': 'ridge',
        'Lasso Regression': 'lasso',
        'Support Vector Regression': 'svr',
        'Neural Network': 'neural_network'
    }
    
    # Model selection
    selected_models = st.sidebar.multiselect(
        "Select ML Models",
        list(available_models.keys()),
        default=['Random Forest', 'Gradient Boosting', 'Linear Regression'],
        help="Choose which models to train and compare"
    )
    
    # Training parameters
    test_size = st.sidebar.slider(
        "Test Size (%)", 
        min_value=10, 
        max_value=50, 
        value=20, 
        step=5,
        help="Percentage of data to use for testing"
    ) / 100
    
    prediction_horizon = st.sidebar.selectbox(
        "Prediction Horizon",
        [1, 3, 6, 12, 24],
        index=0,
        help="How many periods ahead to predict"
    )
    
    # Feature Engineering Section
    st.markdown('<h3 class="sub-header">🔧 Feature Engineering</h3>', unsafe_allow_html=True)
    
    if st.button("Generate Features", type="primary"):
        if len(data) < 50:
            st.warning("⚠️ Limited data available. Consider selecting a longer time period for better ML performance.")
        
        with st.spinner("Generating features..."):
            try:
                # Initialize feature engineer
                engineer = FeatureEngineer()
                
                # Create features with reduced complexity for smaller datasets
                if len(data) < 100:
                    # Use smaller rolling windows for limited data
                    featured_data = engineer.create_basic_features(data)
                else:
                    featured_data = engineer.create_all_features(
                        data, 
                        target_type='price', 
                        prediction_horizon=prediction_horizon
                    )
                
                st.session_state.featured_data = featured_data
                st.session_state.features_generated = True
                
                # Show feature info
                feature_cols = [col for col in featured_data.columns 
                               if col not in ['Datetime', 'datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume']]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"✅ Generated {len(feature_cols)} features")
                    st.write(f"📊 Data shape: {featured_data.shape}")
                
                with col2:
                    st.markdown("**Feature Categories:**")
                    tech_features = [f for f in feature_cols if any(x in f.lower() for x in ['sma', 'ema', 'rsi', 'macd', 'bb'])]
                    price_features = [f for f in feature_cols if any(x in f.lower() for x in ['return', 'volatility', 'high_low'])]
                    time_features = [f for f in feature_cols if any(x in f.lower() for x in ['hour', 'day', 'month', 'dayofweek'])]
                    lag_features = [f for f in feature_cols if 'lag' in f.lower()]
                    
                    st.write(f"• Technical: {len(tech_features)}")
                    st.write(f"• Price-based: {len(price_features)}")
                    st.write(f"• Time-based: {len(time_features)}")
                    st.write(f"• Lag features: {len(lag_features)}")
                
                # Show feature preview
                if len(featured_data) > 0:
                    st.markdown("**Feature Preview:**")
                    preview_cols = feature_cols[:10] if len(feature_cols) > 10 else feature_cols
                    st.dataframe(featured_data[preview_cols].tail(5), use_container_width=True)
                else:
                    st.error("⚠️ No valid data after feature engineering. Try a longer time period.")
                    
            except Exception as e:
                st.error(f"Error generating features: {str(e)}")
                st.write("**Debug info:**")
                st.write(f"Data shape: {data.shape}")
                st.write(f"Columns: {data.columns.tolist()}")
    
    # Model Training Section
    if st.session_state.get('features_generated', False) and len(st.session_state.featured_data) > 0:
        st.markdown('<h3 class="sub-header">🏋️ Model Training</h3>', unsafe_allow_html=True)
        
        if st.button("Train Selected Models", type="primary"):
            if not selected_models:
                st.error("Please select at least one model to train.")
            else:
                featured_data = st.session_state.featured_data
                
                with st.spinner(f"Training {len(selected_models)} models..."):
                    try:
                        # Initialize ML predictor
                        predictor = CryptoPricePredictor()
                        
                        # Prepare features
                        X, y, feature_names = predictor.prepare_features(featured_data)
                        
                        if len(X) == 0:
                            st.error("No valid training data available after preprocessing.")
                        else:
                            from sklearn.model_selection import train_test_split
                            
                            # Split data
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=test_size, random_state=42
                            )
                            
                            # Train selected models
                            results = []
                            progress_bar = st.progress(0)
                            
                            for i, model_display_name in enumerate(selected_models):
                                model_name = available_models[model_display_name]
                                
                                progress_bar.progress((i + 1) / len(selected_models))
                                st.write(f"Training {model_display_name}...")
                                
                                try:
                                    # Train individual model
                                    result = predictor.train_single_model(
                                        model_name, X_train, y_train, X_test, y_test, scale_features=True
                                    )
                                    result['display_name'] = model_display_name
                                    results.append(result)
                                    
                                except Exception as model_error:
                                    st.warning(f"Failed to train {model_display_name}: {str(model_error)}")
                                    continue
                            
                            if results:
                                # Store results
                                results_df = pd.DataFrame(results)
                                st.session_state.model_results = results_df
                                st.session_state.models_trained = True
                                st.session_state.predictor = predictor
                                
                                st.success(f"✅ Successfully trained {len(results)} models!")
                            else:
                                st.error("No models were successfully trained.")
                                
                    except Exception as e:
                        st.error(f"Error during model training: {str(e)}")
    
    # Model Results Section
    if st.session_state.get('models_trained', False):
        st.markdown('<h3 class="sub-header">📊 Model Results</h3>', unsafe_allow_html=True)
        
        results_df = st.session_state.model_results
        
        # Sort by performance (best RMSE first)
        results_df = results_df.sort_values('test_rmse')
        
        # Display performance metrics
        st.markdown("**📈 Model Performance Comparison:**")
        
        # Create performance comparison chart
        fig_comparison = go.Figure()
        
        # RMSE comparison
        fig_comparison.add_trace(go.Bar(
            x=results_df['display_name'],
            y=results_df['test_rmse'],
            name='Test RMSE',
            marker_color='lightcoral',
            text=results_df['test_rmse'].round(4),
            textposition='auto'
        ))
        
        fig_comparison.update_layout(
            title='Model Performance Comparison (Lower RMSE = Better)',
            xaxis_title='Model',
            yaxis_title='RMSE',
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # R² Score comparison
        fig_r2 = go.Figure()
        
        fig_r2.add_trace(go.Bar(
            x=results_df['display_name'],
            y=results_df['test_r2'],
            name='Test R²',
            marker_color='lightblue',
            text=results_df['test_r2'].round(4),
            textposition='auto'
        ))
        
        fig_r2.update_layout(
            title='Model R² Score Comparison (Higher = Better)',
            xaxis_title='Model',
            yaxis_title='R² Score',
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_r2, use_container_width=True)
        
        # Detailed results table
        st.markdown("**📋 Detailed Results:**")
        display_cols = ['display_name', 'test_rmse', 'test_r2', 'test_mae', 'train_rmse', 'train_r2']
        display_df = results_df[display_cols].copy()
        display_df.columns = ['Model', 'Test RMSE', 'Test R²', 'Test MAE', 'Train RMSE', 'Train R²']
        
        # Format numerical columns
        for col in ['Test RMSE', 'Test R²', 'Test MAE', 'Train RMSE', 'Train R²']:
            display_df[col] = display_df[col].round(4)
        
        st.dataframe(display_df, use_container_width=True)
        
        # Best model highlight
        best_model = results_df.iloc[0]
        st.success(f"🏆 **Best Model**: {best_model['display_name']} (RMSE: {best_model['test_rmse']:.4f}, R²: {best_model['test_r2']:.4f})")
        
        # Model interpretation
        if best_model['test_r2'] > 0.8:
            interpretation = "🎯 **Excellent** - Model shows strong predictive power"
        elif best_model['test_r2'] > 0.6:
            interpretation = "✅ **Good** - Model has reasonable predictive ability"
        elif best_model['test_r2'] > 0.3:
            interpretation = "⚠️ **Fair** - Model has limited predictive power"
        else:
            interpretation = "❌ **Poor** - Model may not be reliable for predictions"
        
        st.info(interpretation)
        
        # Predictions Section
        st.markdown('<h3 class="sub-header">🔮 Price Predictions</h3>', unsafe_allow_html=True)
        
        try:
            # Get latest prediction
            predictor = st.session_state.predictor
            featured_data = st.session_state.featured_data
            
            # Prepare features for prediction
            X, y, feature_names = predictor.prepare_features(featured_data)
            
            if len(X) > 0:
                # Get the best model
                best_model_name = best_model['model_name']
                best_model_obj = predictor.models[best_model_name]
                
                # Make prediction on the latest data point
                latest_X = X[-1:]
                prediction = best_model_obj.predict(latest_X)[0]
                
                current_price = data['close'].iloc[-1]
                price_change = prediction - current_price
                price_change_pct = (price_change / current_price) * 100
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Current Price", f"${current_price:.2f}")
                
                with col2:
                    st.metric("Predicted Price", f"${prediction:.2f}")
                
                with col3:
                    st.metric("Expected Change", f"{price_change_pct:.2f}%", f"${price_change:.2f}")
                
                with col4:
                    direction = "📈 Bullish" if price_change > 0 else "📉 Bearish" if price_change < 0 else "➡️ Neutral"
                    st.metric("Direction", direction)
                
                # Prediction confidence
                confidence_level = "High" if best_model['test_r2'] > 0.7 else "Medium" if best_model['test_r2'] > 0.4 else "Low"
                st.write(f"**Prediction Confidence**: {confidence_level} (R² = {best_model['test_r2']:.3f})")
                
                # Model used
                st.write(f"**Model Used**: {best_model['display_name']}")
                st.write(f"**Prediction Horizon**: {prediction_horizon} period(s) ahead")
                
        except Exception as e:
            st.error(f"Error making predictions: {str(e)}")

else:
    st.info("👆 Click 'Load Data' to start analyzing cryptocurrency data!")
    
    # Show sample data structure
    st.markdown('<h2 class="sub-header">📖 About This App</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    This cryptocurrency price predictor app provides:
    
    1. **Real-time Data Loading**: Fetch historical cryptocurrency data from Yahoo Finance
    2. **Interactive Visualizations**: Candlestick charts, volume analysis, and trend analysis
    3. **Price Statistics**: Comprehensive statistics about price and volume
    4. **Trend Analysis**: Moving averages and trend interpretation
    
    **Supported Cryptocurrencies**: BTC, ETH, ADA, DOT, SOL, MATIC, AVAX, LINK
    
    **How to use:**
    - Select a cryptocurrency from the sidebar
    - Choose your preferred time period and interval
    - Click 'Load Data' to fetch historical data
    - Explore the charts and statistics
    """)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>Built with ❤️ using Streamlit | Cryptocurrency Price Predictor</div>", unsafe_allow_html=True)
