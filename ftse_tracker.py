import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import warnings
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="FTSE 100 Tracker & Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# FTSE 100 constituents with Yahoo Finance tickers (.L suffix)
FTSE_100 = {
    "AAL": "Anglo American",
    "ABF": "Associated British Foods",
    "ADM": "Admiral Group",
    "AHT": "Ashtead Group",
    "ANTO": "Antofagasta",
    "AZN": "AstraZeneca",
    "BA": "BAE Systems",
    "BARC": "Barclays",
    "BATS": "British American Tobacco",
    "BP": "BP",
    "BRBY": "Burberry",
    "BT": "BT Group",
    "CCH": "Coca-Cola HBC",
    "CNA": "Centrica",
    "CPG": "Compass Group",
    "DGE": "Diageo",
    "DPLM": "Diploma",
    "EZJ": "easyJet",
    "FLTR": "Flutter Entertainment",
    "FRES": "Fresnillo",
    "GSK": "GlaxoSmithKline",
    "HL": "Hargreaves Lansdown",
    "HSBA": "HSBC",
    "IAG": "International Airlines Group",
    "IMB": "Imperial Brands",
    "ITRK": "Intertek",
    "JMAT": "Johnson Matthey",
    "LAND": "Land Securities",
    "LEG": "Legal & General",
    "LLOY": "Lloyds Banking Group",
    "MNDI": "Mondi",
    "MRO": "Melrose Industries",
    "NG": "National Grid",
    "NXT": "Next",
    "PSON": "Pearson",
    "PSX": "Phoenix Group",
    "REL": "Relx",
    "RIO": "Rio Tinto",
    "RKT": "Reckitt Benckiser",
    "RMV": "Rightmove",
    "RR": "Rolls-Royce",
    "RTO": "Rentokil Initial",
    "SBRY": "Sainsbury's",
    "SDR": "Schroders",
    "SHEL": "Shell",
    "SMIN": "Smiths Group",
    "SN": "Smith & Nephew",
    "SPX": "Spirax-Sarco Engineering",
    "SSE": "SSE",
    "STAN": "Standard Chartered",
    "STJ": "St James's Place",
    "SVT": "Severn Trent",
    "TATE": "Tate & Lyle",
    "TSCO": "Tesco",
    "TW": "Taylor Wimpey",
    "ULVR": "Unilever",
    "VOD": "Vodafone",
    "WEIR": "Weir Group",
    "WPP": "WPP"
}

# ============================================================================
# CACHED DATA FETCHING
# ============================================================================
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_stock_data(ticker, period="1y"):
    """Fetch historical data for a single stock."""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty:
            return None
        return df
    except Exception:
        return None


@st.cache_data(ttl=3600)
def fetch_all_stocks(tickers, period="1y"):
    """Fetch data for all stocks in parallel."""
    results = {}
    with st.spinner(f"Fetching data for {len(tickers)} stocks..."):
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_ticker = {
                executor.submit(fetch_stock_data, f"{ticker}.L", period): ticker
                for ticker in tickers
            }
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    data = future.result()
                    if data is not None and not data.empty:
                        results[ticker] = data
                except Exception:
                    pass
    return results


# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================
def calculate_rsi(data, window=14):
    """Calculate RSI indicator."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD indicator."""
    exp_fast = data.ewm(span=fast, adjust=False).mean()
    exp_slow = data.ewm(span=slow, adjust=False).mean()
    macd_line = exp_fast - exp_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands."""
    sma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, sma, lower


def calculate_indicators(df):
    """Calculate all technical indicators for a stock."""
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    # Moving averages
    df['SMA_5'] = close.rolling(5).mean()
    df['SMA_10'] = close.rolling(10).mean()
    df['SMA_20'] = close.rolling(20).mean()
    df['SMA_50'] = close.rolling(50).mean()

    # RSI
    df['RSI'] = calculate_rsi(close)

    # MACD
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(close)

    # Bollinger Bands
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(close)

    # Volatility (20-day)
    df['Volatility'] = close.pct_change().rolling(20).std() * np.sqrt(252)

    # Price momentum
    df['Momentum_1d'] = close.pct_change()
    df['Momentum_5d'] = close.pct_change(5)
    df['Momentum_10d'] = close.pct_change(10)
    df['Momentum_20d'] = close.pct_change(20)

    # Volume indicators
    df['Volume_SMA'] = volume.rolling(20).mean()
    df['Volume_Ratio'] = volume / df['Volume_SMA']

    # ATR (Average True Range)
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()

    return df


# ============================================================================
# PREDICTION MODEL
# ============================================================================
def prepare_features(df, target_col='Close', lookback=20):
    """Prepare features for ML model."""
    # Use technical indicators as features
    feature_cols = [
        'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'BB_Upper', 'BB_Middle', 'BB_Lower',
        'Volatility', 'Momentum_1d', 'Momentum_5d',
        'Momentum_10d', 'Momentum_20d', 'Volume_Ratio', 'ATR'
    ]

    # Drop rows with NaN
    df_clean = df[feature_cols + [target_col]].dropna()

    if len(df_clean) < lookback + 5:
        return None, None, None, None

    X = df_clean[feature_cols].values
    y = df_clean[target_col].values

    # Use last 'lookback' days for training, predict next day
    X_train = X[:-1]
    y_train = y[1:]  # Predict next day's price

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    return X_train_scaled, y_train, scaler, df_clean


def predict_next_price(df):
    """Predict next day's closing price using Linear Regression."""
    feature_cols = [
        'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'BB_Upper', 'BB_Middle', 'BB_Lower',
        'Volatility', 'Momentum_1d', 'Momentum_5d',
        'Momentum_10d', 'Momentum_20d', 'Volume_Ratio', 'ATR'
    ]

    df_clean = df[feature_cols + ['Close']].dropna()

    if len(df_clean) < 30:
        return None, None, None

    # Features and target
    X = df_clean[feature_cols].values
    y = df_clean['Close'].values

    # Use all but last row for training, predict last row
    X_train = X[:-1]
    y_train = y[1:]  # Predict next day

    if len(X_train) < 10:
        return None, None, None

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Predict next day using the most recent data
    X_last = X[-1:].reshape(1, -1)
    X_last_scaled = scaler.transform(X_last)
    predicted_price = model.predict(X_last_scaled)[0]

    current_price = df['Close'].iloc[-1]
    confidence = model.score(X_train_scaled, y_train)

    return predicted_price, current_price, confidence


# ============================================================================
# STOCK ANALYSIS
# ============================================================================
def analyze_stock(ticker, df):
    """Analyze a single stock and return metrics."""
    if df is None or df.empty or len(df) < 30:
        return None

    df = calculate_indicators(df.copy())
    last_row = df.iloc[-1]

    # Current price and metrics
    current_price = df['Close'].iloc[-1]
    prev_close = df['Close'].iloc[-2] if len(df) > 1 else current_price
    daily_change = ((current_price - prev_close) / prev_close) * 100

    # Price relative to moving averages
    sma_20 = last_row['SMA_20']
    sma_50 = last_row['SMA_50']
    price_vs_sma20 = ((current_price - sma_20) / sma_20) * 100 if sma_20 else 0
    price_vs_sma50 = ((current_price - sma_50) / sma_50) * 100 if sma_50 else 0

    # RSI signal
    rsi = last_row['RSI']
    rsi_signal = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"

    # MACD signal
    macd = last_row['MACD']
    macd_signal = last_row['MACD_Signal']
    macd_cross = "Bullish" if macd > macd_signal else "Bearish"

    # Bollinger Band position
    bb_upper = last_row['BB_Upper']
    bb_lower = last_row['BB_Lower']
    bb_middle = last_row['BB_Middle']
    if bb_upper and bb_lower:
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) != 0 else 0.5
        bb_signal = "Overbought" if bb_position > 0.8 else "Oversold" if bb_position < 0.2 else "Neutral"
    else:
        bb_position = 0.5
        bb_signal = "Neutral"

    # Momentum
    mom_5d = last_row['Momentum_5d'] * 100 if not pd.isna(last_row['Momentum_5d']) else 0
    mom_20d = last_row['Momentum_20d'] * 100 if not pd.isna(last_row['Momentum_20d']) else 0

    # Volatility
    volatility = last_row['Volatility'] * 100 if not pd.isna(last_row['Volatility']) else 0

    # ML Prediction
    predicted_price, current_price_ml, confidence = predict_next_price(df)

    if predicted_price is not None:
        predicted_change = ((predicted_price - current_price_ml) / current_price_ml) * 100
        ml_signal = "BUY" if predicted_change > 2 else "SELL" if predicted_change < -2 else "HOLD"
    else:
        predicted_change = 0
        ml_signal = "HOLD"
        confidence = 0

    # Combined score (0-100, higher = stronger buy signal)
    # Weighted combination of indicators
    score = 50

    # RSI: oversold = buy signal
    if rsi < 30:
        score += 15
    elif rsi > 70:
        score -= 15

    # MACD: bullish cross = buy
    if macd_cross == "Bullish":
        score += 10
    else:
        score -= 10

    # Bollinger: oversold = buy
    if bb_signal == "Oversold":
        score += 10
    elif bb_signal == "Overbought":
        score -= 10

    # Price vs SMA: below SMA50 = potential buy (mean reversion)
    if price_vs_sma50 < -10:
        score += 10
    elif price_vs_sma50 > 10:
        score -= 10

    # Momentum: positive momentum = buy
    if mom_5d > 2:
        score += 5
    elif mom_5d < -2:
        score -= 5

    # ML prediction
    if ml_signal == "BUY":
        score += 15
    elif ml_signal == "SELL":
        score -= 15

    # Clamp score
    score = max(0, min(100, score))

    # Recommendation
    if score >= 65:
        recommendation = "STRONG BUY"
    elif score >= 55:
        recommendation = "BUY"
    elif score >= 45:
        recommendation = "HOLD"
    elif score >= 35:
        recommendation = "SELL"
    else:
        recommendation = "STRONG SELL"

    return {
        'ticker': ticker,
        'current_price': current_price,
        'daily_change': daily_change,
        'sma_20': sma_20,
        'sma_50': sma_50,
        'price_vs_sma20': price_vs_sma20,
        'price_vs_sma50': price_vs_sma50,
        'rsi': rsi,
        'rsi_signal': rsi_signal,
        'macd_cross': macd_cross,
        'bb_signal': bb_signal,
        'bb_position': bb_position,
        'mom_5d': mom_5d,
        'mom_20d': mom_20d,
        'volatility': volatility,
        'predicted_price': predicted_price,
        'predicted_change': predicted_change,
        'ml_signal': ml_signal,
        'ml_confidence': confidence,
        'score': score,
        'recommendation': recommendation,
        'data': df
    }


# ============================================================================
# UI COMPONENTS
# ============================================================================
def render_sidebar():
    """Render the sidebar with controls."""
    st.sidebar.title("📊 FTSE 100 Tracker")
    st.sidebar.markdown("---")

    # Stock selection
    available_tickers = list(FTSE_100.keys())
    selected_tickers = st.sidebar.multiselect(
        "Select stocks to track",
        options=available_tickers,
        default=available_tickers[:20],
        help="Choose which FTSE 100 stocks to analyze"
    )

    if not selected_tickers:
        st.sidebar.warning("Please select at least one stock")
        selected_tickers = available_tickers[:20]

    # Period selection
    period = st.sidebar.selectbox(
        "Data period",
        options=["3mo", "6mo", "1y", "2y"],
        index=2,
        help="Amount of historical data to use"
    )

    # Refresh button
    refresh = st.sidebar.button("🔄 Refresh Data", type="primary")

    # Show number of stocks
    st.sidebar.markdown("---")
    st.sidebar.info(f"Tracking {len(selected_tickers)} stocks")

    # Legend
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Recommendation Guide**")
    st.sidebar.markdown("🟢 **STRONG BUY** (score ≥ 65)")
    st.sidebar.markdown("🟡 **BUY** (score 55-64)")
    st.sidebar.markdown("⚪ **HOLD** (score 45-54)")
    st.sidebar.markdown("🟠 **SELL** (score 35-44)")
    st.sidebar.markdown("🔴 **STRONG SELL** (score < 35)")

    return selected_tickers, period, refresh


def render_dashboard(results):
    """Render the main dashboard."""
    if not results:
        st.warning("No data available. Please try refreshing.")
        return

    # Convert results to DataFrame for display
    rows = []
    for ticker, data in results.items():
        if data:
            rows.append({
                'Ticker': ticker,
                'Company': FTSE_100.get(ticker, ticker),
                'Price': f"£{data['current_price']:.2f}" if data['current_price'] else "N/A",
                'Change %': f"{data['daily_change']:.2f}%" if data['daily_change'] else "N/A",
                'RSI': f"{data['rsi']:.1f}" if data['rsi'] else "N/A",
                'Score': data['score'],
                'Recommendation': data['recommendation'],
                'ML Signal': data['ml_signal'],
                'Predicted Change': f"{data['predicted_change']:.2f}%" if data['predicted_change'] else "N/A",
                'Volatility': f"{data['volatility']:.1f}%" if data['volatility'] else "N/A",
                '_data': data
            })

    if not rows:
        st.warning("No valid data available")
        return

    df_display = pd.DataFrame(rows)

    # Sort by score (highest first = best buy)
    df_display = df_display.sort_values('Score', ascending=False)

    # ===== TOP RECOMMENDATIONS =====
    st.markdown("## 🎯 Top Recommendations")

    col1, col2, col3, col4 = st.columns(4)

    # Top BUY recommendations
    top_buy = df_display[df_display['Recommendation'].isin(['STRONG BUY', 'BUY'])].head(3)
    top_sell = df_display[df_display['Recommendation'].isin(['STRONG SELL', 'SELL'])].head(3)

    with col1:
        st.metric("📈 Top Pick", top_buy.iloc[0]['Ticker'] if len(top_buy) > 0 else "N/A",
                  f"{top_buy.iloc[0]['Predicted Change']}" if len(top_buy) > 0 else "N/A")
        if len(top_buy) > 0:
            st.caption(f"{top_buy.iloc[0]['Company']} — {top_buy.iloc[0]['Recommendation']}")

    with col2:
        st.metric("📊 2nd Pick", top_buy.iloc[1]['Ticker'] if len(top_buy) > 1 else "N/A",
                  f"{top_buy.iloc[1]['Predicted Change']}" if len(top_buy) > 1 else "N/A")
        if len(top_buy) > 1:
            st.caption(f"{top_buy.iloc[1]['Company']} — {top_buy.iloc[1]['Recommendation']}")

    with col3:
        st.metric("📉 3rd Pick", top_buy.iloc[2]['Ticker'] if len(top_buy) > 2 else "N/A",
                  f"{top_buy.iloc[2]['Predicted Change']}" if len(top_buy) > 2 else "N/A")
        if len(top_buy) > 2:
            st.caption(f"{top_buy.iloc[2]['Company']} — {top_buy.iloc[2]['Recommendation']}")

    with col4:
        if len(top_sell) > 0:
            st.metric("⚠️ Top Sell", top_sell.iloc[0]['Ticker'],
                      f"{top_sell.iloc[0]['Predicted Change']}", delta_color="inverse")
            st.caption(f"{top_sell.iloc[0]['Company']} — {top_sell.iloc[0]['Recommendation']}")
        else:
            st.metric("⚠️ Top Sell", "N/A", "N/A")

    st.markdown("---")

    # ===== STOCK LIST =====
    st.markdown("## 📋 All Stocks")

    # Color mapping for recommendations
    def color_recommendation(rec):
        colors = {
            'STRONG BUY': 'background-color: #00ff00; color: black;',
            'BUY': 'background-color: #90ee90; color: black;',
            'HOLD': 'background-color: #ffff00; color: black;',
            'SELL': 'background-color: #ffa500; color: black;',
            'STRONG SELL': 'background-color: #ff0000; color: white;'
        }
        return colors.get(rec, '')

    # Display table with styling
    display_cols = ['Ticker', 'Company', 'Price', 'Change %', 'RSI', 'Score', 'Recommendation', 'ML Signal',
                    'Predicted Change', 'Volatility']
    df_table = df_display[display_cols].copy()

    # Apply color to recommendation column
    def highlight_rec(row):
        return [color_recommendation(row['Recommendation'])] * len(row)

    # Display as a styled dataframe
    st.dataframe(
        df_table.style.apply(highlight_rec, axis=1),
        use_container_width=True,
        height=400
    )

    st.markdown("---")

    # ===== DETAILED CHARTS =====
    st.markdown("## 📊 Detailed Analysis")

    # Let user select a stock to view details
    selected_ticker = st.selectbox(
        "Select a stock for detailed analysis",
        options=df_display['Ticker'].tolist(),
        format_func=lambda x: f"{x} - {FTSE_100.get(x, x)}"
    )

    if selected_ticker:
        selected_data = df_display[df_display['Ticker'] == selected_ticker].iloc[0]
        render_stock_detail(selected_ticker, selected_data['_data'])


def render_stock_detail(ticker, data):
    """Render detailed view for a single stock."""
    if not data or 'data' not in data:
        st.warning("No detailed data available")
        return

    df = data['data']

    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Current Price", f"£{data['current_price']:.2f}" if data['current_price'] else "N/A",
                  f"{data['daily_change']:.2f}%")

    with col2:
        st.metric("RSI", f"{data['rsi']:.1f}" if data['rsi'] else "N/A",
                  data['rsi_signal'] if data['rsi_signal'] else "")

    with col3:
        st.metric("MACD", data['macd_cross'] if data['macd_cross'] else "N/A")

    with col4:
        st.metric("Bollinger", data['bb_signal'] if data['bb_signal'] else "N/A")

    with col5:
        st.metric("Recommendation", data['recommendation'],
                  f"Score: {data['score']:.0f}/100")

    # Price chart with indicators
    st.markdown("### Price & Technical Indicators")

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=("Price with SMA & Bollinger Bands", "RSI", "MACD")
    )

    # Price
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Close'], name="Close Price", line=dict(color='blue')),
        row=1, col=1
    )

    # SMAs
    fig.add_trace(
        go.Scatter(x=df.index, y=df['SMA_20'], name="SMA 20", line=dict(color='orange', dash='dash')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['SMA_50'], name="SMA 50", line=dict(color='red', dash='dash')),
        row=1, col=1
    )

    # Bollinger Bands
    fig.add_trace(
        go.Scatter(x=df.index, y=df['BB_Upper'], name="BB Upper", line=dict(color='gray', width=1), showlegend=True),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['BB_Lower'], name="BB Lower", line=dict(color='gray', width=1), showlegend=True,
                   fill='tonexty', fillcolor='rgba(128,128,128,0.1)'),
        row=1, col=1
    )

    # RSI
    fig.add_trace(
        go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color='purple')),
        row=2, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # MACD
    fig.add_trace(
        go.Scatter(x=df.index, y=df['MACD'], name="MACD", line=dict(color='blue')),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['MACD_Signal'], name="Signal", line=dict(color='orange')),
        row=3, col=1
    )
    # MACD Histogram
    colors = ['green' if x >= 0 else 'red' for x in df['MACD_Hist'].dropna()]
    fig.add_trace(
        go.Bar(x=df.index, y=df['MACD_Hist'], name="Histogram", marker_color=colors),
        row=3, col=1
    )

    fig.update_layout(height=700, showlegend=True, hovermode='x unified')
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Price (£)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # Additional metrics
    st.markdown("### 📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Price vs SMA20", f"{data['price_vs_sma20']:.2f}%" if data['price_vs_sma20'] else "N/A")
    with col2:
        st.metric("Price vs SMA50", f"{data['price_vs_sma50']:.2f}%" if data['price_vs_sma50'] else "N/A")
    with col3:
        st.metric("5-Day Momentum", f"{data['mom_5d']:.2f}%" if data['mom_5d'] else "N/A")
    with col4:
        st.metric("20-Day Momentum", f"{data['mom_20d']:.2f}%" if data['mom_20d'] else "N/A")

    # ML Prediction
    st.markdown("### 🤖 ML Prediction")
    if data['predicted_price']:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Price", f"£{data['predicted_price']:.2f}")
        with col2:
            st.metric("Expected Change", f"{data['predicted_change']:.2f}%")
        with col3:
            st.metric("Model Confidence", f"{data['ml_confidence']*100:.1f}%" if data['ml_confidence'] else "N/A")
        st.caption(
            "ML Signal: Based on Linear Regression model trained on technical indicators. Confidence is the R² score of the model.")
    else:
        st.info("Insufficient data for ML prediction (need at least 30 days of data).")


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    """Main application entry point."""
    # Sidebar
    selected_tickers, period, refresh = render_sidebar()

    # Main content
    st.title("📈 FTSE 100 Stock Tracker & Predictor")
    st.caption("Track FTSE 100 shares and get AI-powered buy/sell recommendations")

    # Fetch data
    if 'data_cache' not in st.session_state or refresh:
        with st.spinner("Fetching latest data..."):
            data = fetch_all_stocks(selected_tickers, period)
            st.session_state.data_cache = data
            st.session_state.last_refresh = datetime.now()
    else:
        data = st.session_state.data_cache

    # Show last refresh time
    if 'last_refresh' in st.session_state:
        st.caption(f"Last updated: {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}")

    # Analyze stocks
    if data:
        with st.spinner("Analyzing stocks..."):
            results = {}
            progress_bar = st.progress(0)
            tickers = list(data.keys())

            for i, ticker in enumerate(tickers):
                df = data[ticker]
                if df is not None and not df.empty:
                    analysis = analyze_stock(ticker, df)
                    if analysis:
                        results[ticker] = analysis
                progress_bar.progress((i + 1) / len(tickers))

            progress_bar.empty()

            # Render dashboard
            render_dashboard(results)
    else:
        st.error("Failed to fetch data. Please check your internet connection and try again.")


if __name__ == "__main__":
    main()