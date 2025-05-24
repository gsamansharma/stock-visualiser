import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta
import requests
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from dotenv import load_dotenv
import os
# Set page configuration
st.set_page_config(page_title="RL Stock Trading", page_icon="📈", layout="wide")

# --- DATA FETCHING AND PREPROCESSING ---
@st.cache_data
def load_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            return None
        # Standardize column names
        data.columns = [col.lower() if not isinstance(col, tuple) else col[0].lower() for col in data.columns]
        return data
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return None

def preprocess_data(data):
    df = data.copy()
    
    # Calculate technical indicators
    # Moving averages
    df['sma20'] = df['close'].rolling(window=20).mean()
    df['sma50'] = df['close'].rolling(window=50).mean()
    
    # Bollinger Bands
    df['std20'] = df['close'].rolling(window=20).std()
    df['upperband'] = df['sma20'] + (df['std20'] * 2)
    df['lowerband'] = df['sma20'] - (df['std20'] * 2)
    
    # MACD
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Add daily returns
    df['daily_return'] = df['close'].pct_change()
    
    # Drop rows with NaN values
    df.dropna(inplace=True)
    
    return df

# --- VISUALIZATION FUNCTIONS ---
def plot_candlestick(data, ticker):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, subplot_titles=('', 'Volume'),
                        row_width=[0.2, 0.7])
    
    # Determine colors for candlesticks
    colors = ['red' if data['close'].iloc[i] < data['open'].iloc[i] else 'green' for i in range(len(data))]
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name=f"{ticker} Price"
        ),
        row=1, col=1
    )
    
    # Add volume chart
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['volume'],
            marker_color=colors,
            name='Volume'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f"{ticker} Stock Price and Volume",
        xaxis_title="Date",
        yaxis_title="Price",
        height=600,
        xaxis_rangeslider_visible=False,
        template="plotly_white"
    )
    
    return fig

def plot_technical_indicators(data, ticker):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05,
                        subplot_titles=('Price with MA and Bollinger Bands', 'MACD', 'RSI'),
                        row_heights=[0.5, 0.25, 0.25])
    
    # Price + MA + Bollinger Bands
    fig.add_trace(go.Scatter(x=data.index, y=data['close'], name='Close', line=dict(color='black')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['sma20'], name='SMA20', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['sma50'], name='SMA50', line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['upperband'], name='Upper BB', line=dict(color='rgba(0,128,0,0.3)')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['lowerband'], name='Lower BB', line=dict(color='rgba(0,128,0,0.3)'), fill='tonexty', fillcolor='rgba(0,128,0,0.1)'), row=1, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=data.index, y=data['macd'], name='MACD', line=dict(color='blue')), row=2, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['signal'], name='Signal', line=dict(color='red')), row=2, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=data.index, y=data['rsi'], name='RSI', line=dict(color='purple')), row=3, col=1)
    fig.add_trace(go.Scatter(x=[data.index[0], data.index[-1]], y=[70, 70], name='Overbought', line=dict(color='red', dash='dash')), row=3, col=1)
    fig.add_trace(go.Scatter(x=[data.index[0], data.index[-1]], y=[30, 30], name='Oversold', line=dict(color='green', dash='dash')), row=3, col=1)
    
    fig.update_layout(height=800, template="plotly_white", showlegend=True)
    
    return fig

# --- NEW REINFORCEMENT LEARNING ENVIRONMENT ---
class StockTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=1000):
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.initial_balance = initial_balance
        self.current_step = 0
        
        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6,), dtype=np.float32
        )  # Normalized features: Open, High, Low, Close, Volume, Balance
    
    def reset(self, seed=None, options=None):
        # Reset the environment to the initial state
        super().reset(seed=seed)  # Required by gymnasium
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = 0
        self.net_worth_history = [self.initial_balance]
        self.trades = []
        return self._next_observation(), {}  # Return observation and info (empty dict)
    
    def _next_observation(self):
        # Get the current stock data and normalize it
        row = self.df.iloc[self.current_step]
        max_price = max(self.df['high'].max(), 1)  # Avoid division by zero
        max_volume = max(self.df['volume'].max(), 1)  # Avoid division by zero
        
        obs = np.array([
            row['open'] / max_price,  # Normalize Open
            row['high'] / max_price,  # Normalize High
            row['low'] / max_price,   # Normalize Low
            row['close'] / max_price, # Normalize Close
            row['volume'] / max_volume, # Normalize Volume
            self.balance / self.initial_balance  # Normalize Balance
        ], dtype=np.float32)
        
        return obs
    
    def step(self, action):
        # Execute one step in the environment
        current_price = self.df.iloc[self.current_step]['close']
        reward = 0
        done = False
        truncated = False
        
        # Execute action
        if action == 1:  # Buy
            if self.balance >= current_price:
                self.shares_held += 1
                self.balance -= current_price
                self.trades.append({'step': self.current_step, 'type': 'buy', 'price': current_price})
        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.shares_held -= 1
                self.balance += current_price
                self.trades.append({'step': self.current_step, 'type': 'sell', 'price': current_price})
        
        # Move to the next step
        self.current_step += 1
        
        # Calculate portfolio value
        portfolio_value = self.balance + (self.shares_held * current_price)
        self.net_worth_history.append(portfolio_value)
        
        # Calculate reward (change in portfolio value)
        reward = portfolio_value - self.initial_balance
        
        # Check if the episode is done
        done = self.current_step >= len(self.df) - 1
        
        # Get the next observation
        next_obs = self._next_observation() if not done else None
        
        # Return step information
        info = {
            "net_worth": portfolio_value,
            "balance": self.balance,
            "shares_held": self.shares_held,
            "current_price": current_price
        }
        
        return next_obs, reward, done, truncated, info
    
    def render(self, mode='human'):
        # Display the current state
        print(f"Step: {self.current_step}, Balance: {self.balance}, Shares Held: {self.shares_held}")

@st.cache_data(ttl=3600)
def search_stock_symbol(query):
    try:
        api_key = os.getenv('FMP_API_KEY')
        url = f"https://financialmodelingprep.com/api/v3/search?query={query}&apikey={api_key}"
        print(url)
        response = requests.get(url)
        if response.status_code != 200:
            st.error(f"API request failed with status code {response.status_code}")
            return []
        
        # Parse the response
        data = response.json()
        
        # Handle different response structures
        if isinstance(data, dict) and 'data' in data:
            # Some APIs nest results under a 'data' key
            return data['data']
        elif isinstance(data, list):
            # Direct list of results
            return data
        else:
            st.error(f"Unexpected response format: {type(data)}")
            return []
    except Exception as e:
        st.error(f"Error searching for stock symbol: {e}")
        return []

# --- MAIN APP ---
def main():
    st.title("ProfitPulse: Reinforcement Learning Stock Trading")
    st.write("An AI-powered stock trading platform using reinforcement learning")
    
    

    company_name = st.sidebar.text_input("Enter company name (e.g. Apple, Tesla):")

    if company_name:
        with st.spinner("Searching for stock symbols..."):
            results = search_stock_symbol(company_name)
            print(results)
            if results and isinstance(results, list):
                # Create options list safely
                options = []
                for item in results:
                    if isinstance(item, dict) and 'name' in item and 'symbol' in item:
                        options.append(f"{item['name']} ({item['symbol']})")
                
                if options:
                    selected = st.sidebar.selectbox("Select the correct company:", options)
                    
                    # Find the selected item's symbol
                    selected_item = None
                    for item in results:
                        option_text = f"{item['name']} ({item['symbol']})"
                        if option_text == selected:
                            selected_item = item
                            break
                    
                    if selected_item:
                        selected_symbol = selected_item['symbol']
                        st.session_state.selected_symbol = selected_symbol
                        st.sidebar.success(f"Selected symbol: {selected_symbol}")
                        
                        if st.sidebar.button("Analyze Selected Stock"):
                            st.session_state.page = "Stock Analysis"
                    else:
                        st.sidebar.warning("No matching stocks found")

    st.sidebar.markdown("---")
    st.sidebar.header("Developed By")
    st.sidebar.write("**Deepanshu Jindal**\n21BCS1933")
    st.sidebar.write("**Ankit Panigarhi**\n21BCS2588")
    st.sidebar.write("**Aman Sharma**\n21BCS2322")

 
    st.header("Stock Analysis")
        
        # Check if a stock is selected
    if 'selected_symbol' not in st.session_state:
        st.warning("Please select a stock from the Stock Search page.")
        return
        
    stock_symbol = st.session_state.selected_symbol
        
        # Date range selection
    today = date.today()
    default_start = today - timedelta(days=365)
        
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", default_start)
    with col2:
        end_date = st.date_input("End Date", today)
        
    if start_date >= end_date:
        st.error("Error: End date must be after start date")
        return
        
        # Load and process data
    with st.spinner('Loading data...'):
        data = load_data(stock_symbol, start_date, end_date)
            
        if data is None or data.empty:
            st.error(f"No data found for {stock_symbol} in the specified date range.")
            return
                
        st.success(f"Loaded {len(data)} days of data for {stock_symbol}")
        processed_data = preprocess_data(data)
        
        # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Technical Analysis", "Reinforcement Learning", "Glossary and About"])
        
    with tab1:
        st.subheader(f"{stock_symbol} Stock Overview")
            
        try:
            ticker = yf.Ticker(stock_symbol)
            info = ticker.info
                
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Company Information")
                st.write(f"**Name:** {info.get('longName', 'N/A')}")
                st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                
            with col2:
                st.subheader("Key Statistics")
                st.write(f"**Market Cap:** ${info.get('marketCap', 0):,}")
                st.write(f"**P/E Ratio:** {info.get('trailingPE', 'N/A')}")
                st.write(f"**Dividend Yield:** {info.get('dividendYield', 0)*100:.2f}%")
        except:
            st.warning("Could not fetch company information")
            
        st.subheader("Price Chart")
        fig = plot_candlestick(data, stock_symbol)
        st.plotly_chart(fig, use_container_width=True)
            
        st.subheader("Summary Statistics")
        col1, col2, col3 = st.columns(3)
            
        with col1:
            current_price = data['close'].iloc[-1]
            previous_price = data['close'].iloc[-2]
            price_change = current_price - previous_price
            price_change_pct = (price_change / previous_price) * 100
            st.metric("Current Price", f"${current_price:.2f}", f"{price_change_pct:.2f}%")
            
        with col2:
            high_52w = data['high'].max()
            st.metric("52 Week high", f"${high_52w:.2f}")
            
        with col3:
            low_52w = data['low'].min()
            st.metric("52 Week Low", f"${low_52w:.2f}")
        
    with tab2:
        st.subheader("Technical Analysis")
        tech_fig = plot_technical_indicators(processed_data, stock_symbol)
        st.plotly_chart(tech_fig, use_container_width=True)
            
        st.subheader("Trading Signals")
        last_row = processed_data.iloc[-1]
            
        col1, col2, col3 = st.columns(3)
            
        with col1:
            if last_row['sma20'] > last_row['sma50']:
                st.success("MA Signal: Bullish (SMA20 > SMA50)")
            else:
                st.error("MA Signal: Bearish (SMA20 < SMA50)")
            
        with col2:
            if last_row['rsi'] < 30:
                st.success("RSI Signal: Oversold (Buy)")
            elif last_row['rsi'] > 70:
                st.error("RSI Signal: Overbought (Sell)")
            else:
                st.info("RSI Signal: Neutral")
            
        with col3:
            if last_row['macd'] > last_row['signal']:
                st.success("MACD Signal: Bullish")
            else:
                st.error("MACD Signal: Bearish")
        
    with tab3:
        st.subheader("Reinforcement Learning Stock Trading")
            
            # Training parameters
        col1, col2 = st.columns(2)
            
        with col1:
            initial_balance = st.number_input("Initial Balance ($)", 1000, 100000, 10000, 1000)
            train_test_split = st.slider("Training/Testing Split", 0.5, 0.9, 0.8, 0.05)
            
        with col2:
            total_timesteps = st.number_input("Training Timesteps", 1000, 50000, 10000, 1000)
            model_type = st.selectbox("RL Algorithm", ["PPO", "A2C", "DQN"])
            
            # Split data for training and testing
        split_idx = int(len(processed_data) * train_test_split)
        train_data = processed_data.iloc[:split_idx]
        test_data = processed_data.iloc[split_idx:]
            
            # Train model button
        if st.button("Train RL Model"):
            with st.spinner("Training model... This may take a few minutes."):
                # Create and train the model
                env = StockTradingEnv(train_data, initial_balance=initial_balance)
                env = DummyVecEnv([lambda: env])
                    
                if model_type == "PPO":
                    model = PPO("MlpPolicy", env, verbose=1)
                elif model_type == "A2C":
                    model = A2C("MlpPolicy", env, verbose=1)
                else:  # DQN
                    model = DQN("MlpPolicy", env, verbose=1)
                    
                model.learn(total_timesteps=total_timesteps)
                    
                    # Save the model in session state
                st.session_state.trained_model = model
                st.session_state.model_type = model_type
                    
                st.success("Model training completed!")
            
            # Evaluate model button (only show if model is trained)
        if 'trained_model' in st.session_state:
            st.subheader("Model Evaluation on Test Data")
                
            if st.button("Evaluate Model"):
                with st.spinner("Evaluating model on test data..."):
                    # Create evaluation environment
                    eval_env = StockTradingEnv(test_data, initial_balance=initial_balance)
                        
                        # Run evaluation
                    obs, _ = eval_env.reset()
                    done = False
                    rewards = []
                    actions = []
                    net_worth_history = [initial_balance]
                        
                    while not done:
                        action, _ = st.session_state.trained_model.predict(obs)
                        obs, reward, done, _, info = eval_env.step(action)
                        rewards.append(reward)
                        actions.append(action)
                        net_worth_history.append(info['net_worth'])
                        
                        # Create action labels
                    action_labels = []
                    for action in actions:
                        if action == 0:
                            action_labels.append('HOLD')
                        elif action == 1:
                            action_labels.append('BUY')
                        else:
                            action_labels.append('SELL')
                        
                        # Create evaluation dataframe
                    eval_df = pd.DataFrame({
                        'Date': test_data.index[:len(actions)],
                        'Close': test_data['close'].values[:len(actions)],
                        'Action': action_labels,
                        'NetWorth': net_worth_history[:len(actions)]
                    })
                        
                        # Plot results
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                                        subplot_titles=("Stock Price with Trading Actions", "Portfolio Net Worth"))
                        
                    fig.add_trace(go.Scatter(x=eval_df['Date'], y=eval_df['Close'], mode='lines', name='Stock Price'), row=1, col=1)
                        
                    buy_points = eval_df[eval_df['Action'] == 'BUY']
                    if not buy_points.empty:
                        fig.add_trace(go.Scatter(x=buy_points['Date'], y=buy_points['Close'], mode='markers',
                                                marker=dict(color='green', size=10, symbol='triangle-up'), name='Buy Signal'), row=1, col=1)
                        
                    sell_points = eval_df[eval_df['Action'] == 'SELL']
                    if not sell_points.empty:
                        fig.add_trace(go.Scatter(x=sell_points['Date'], y=sell_points['Close'], mode='markers',
                                                marker=dict(color='red', size=10, symbol='triangle-down'), name='Sell Signal'), row=1, col=1)
                        
                    fig.add_trace(go.Scatter(x=eval_df['Date'], y=eval_df['NetWorth'], mode='lines', name='Portfolio Value'), row=2, col=1)
                        
                    fig.update_layout(height=600, title_text="Trading Strategy Evaluation")
                    st.plotly_chart(fig, use_container_width=True)
                        
                        # Performance metrics
                    final_balance = net_worth_history[-1]
                    total_return = (final_balance - initial_balance) / initial_balance * 100
                        
                    st.subheader("Performance Metrics")
                    col1, col2, col3 = st.columns(3)
                       
                    with col1:
                        st.metric("Final Balance", f"${final_balance:.2f}")
                        
                    with col2:
                        st.metric("Total Return", f"{total_return:.2f}%")
                        
                    with col3:
                        buy_count = action_labels.count('BUY')
                        sell_count = action_labels.count('SELL')
                        st.metric("Total Trades", f"{buy_count + sell_count}")
    with tab4:

        st.subheader("About This App")
        st.write("""
        This application uses Yahoo Finance data to visualize stock prices and technical indicators.
        It implements a Reinforcement Learning model to predict optimal trading actions.
        
        **Key Features:**
        - Real-time Stock Data
        - Technical Analysis with MA, Bollinger Bands, MACD, RSI
        - Reinforcement Learning Trading Agent
        - Interactive Visualizations
    """)
    
        st.subheader("Technical Jargons Explained")
    
        st.write("**P/E Ratio (Price-to-Earnings Ratio)**")
        st.write("""
        The Price-to-Earnings ratio is a valuation metric that compares a company's stock price to its earnings per share (EPS).
        
        **Formula:** P/E Ratio = Current Market Price of a Share / Earnings per Share
        
        A high P/E ratio may indicate that investors expect higher earnings growth in the future, while a low P/E ratio might suggest the stock is undervalued or that the company is facing challenges.
        
        **Example:** If a stock is trading at $100 and its EPS is $5, the P/E ratio is 20, meaning investors are willing to pay $20 for every $1 of earnings.
    """)
    
        st.write("**52-Week High/Low**")
        st.write("""
        The highest and lowest prices at which a stock has traded during the past 52 weeks (one calendar year). 
        This range gives investors an idea of the stock's volatility and price movement over the year.
    """)
    
        st.write("**Moving Averages (MA)**")
        st.write("""
        A calculation used to analyze data points by creating a series of averages of different subsets of the full data set.
        In stock trading, common periods are 20-day (SMA20) and 50-day (SMA50) moving averages.
        
        When a shorter-term MA crosses above a longer-term MA, it's often considered a bullish signal, and vice versa.
    """)
    
        st.write("**Bollinger Bands**")
        st.write("""
        A technical analysis tool defined by a set of trendlines plotted two standard deviations away from a simple moving average of the price.
        
        When prices move close to the upper band, the market might be overbought; when they move close to the lower band, the market might be oversold.
    """)
    
        st.write("**MACD (Moving Average Convergence Divergence)**")
        st.write("""
        A trend-following momentum indicator that shows the relationship between two moving averages of a security's price.
        
        Calculated by subtracting the 26-period EMA from the 12-period EMA, with a 9-period EMA "signal line" plotted on top.
        
        When MACD crosses above the signal line, it's often considered a bullish signal, and vice versa.
    """)
    
        st.write("**RSI (Relative Strength Index)**")
        st.write("""
        A momentum oscillator that measures the speed and change of price movements on a scale from 0 to 100.
        
        Traditionally, RSI values over 70 indicate that a security is overbought (potential sell signal), while values below 30 indicate oversold conditions (potential buy signal).
    """)
    
        st.write("**Disclaimer:**")
        st.write("""
        This app is for educational purposes only and should not be considered financial advice.
    """)

   

# Add Created By section to sidebar
if __name__ == "__main__":
    main()
