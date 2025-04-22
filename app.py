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
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import requests
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

# --- REINFORCEMENT LEARNING ENVIRONMENT ---
class StockTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000, transaction_cost=0.001):
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        
        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        
        # Observation space: normalized price data and account info
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        # Reset the environment to the initial state
        if hasattr(super(), 'reset'):
            super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = 0
        self.net_worth_history = [self.initial_balance]
        self.trades = []
        
        return self._get_state(), {}  # Return observation and info dict
    
    def _get_state(self):
        if self.current_step >= len(self.df):
            return None
            
        current_data = self.df.iloc[self.current_step]
        
        # Create normalized state representation
        state = [
            current_data['close'] / current_data['open'] - 1,  # Price change
            current_data['sma20'] / current_data['close'] - 1,  # SMA20 relative to price
            current_data['sma50'] / current_data['close'] - 1,  # SMA50 relative to price
            current_data['rsi'] / 100,  # RSI (normalized 0-1)
            current_data['macd'] / current_data['close'],  # MACD relative to price
            self.shares_held / 100,  # Shares held (normalized)
            self.balance / self.initial_balance  # Account balance (normalized)
        ]
        
        return np.array(state, dtype=np.float32)
    
    def step(self, action):
        # Execute one step in the environment
        current_price = self.df.iloc[self.current_step]['close']
        
        # Record the current price for visualization
        self.current_price = current_price
        
        # Initialize reward
        reward = 0
        done = False
        truncated = False
        
        # Execute action
        if action == 1:  # Buy
            max_shares_to_buy = self.balance // current_price
            if max_shares_to_buy > 0:
                shares_to_buy = max_shares_to_buy
                cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                self.balance -= cost
                self.shares_held += shares_to_buy
                self.trades.append({'step': self.current_step, 'type': 'buy', 'price': current_price})
        
        elif action == 2:  # Sell
            if self.shares_held > 0:
                proceeds = self.shares_held * current_price * (1 - self.transaction_cost)
                self.balance += proceeds
                self.shares_held = 0
                self.trades.append({'step': self.current_step, 'type': 'sell', 'price': current_price})
        
        # Move to the next step
        self.current_step += 1
        
        # Calculate portfolio value
        if self.current_step < len(self.df):
            next_price = self.df.iloc[self.current_step]['close']
            portfolio_value = self.balance + self.shares_held * next_price
        else:
            portfolio_value = self.balance + self.shares_held * current_price
        
        # Update net worth history
        self.net_worth_history.append(portfolio_value)
        
        # Calculate reward as the change in portfolio value
        reward = (portfolio_value / self.net_worth_history[-2]) - 1
        
        # Check if the episode is done
        done = self.current_step >= len(self.df) - 1
        
        # Get the next state
        next_state = self._get_state()
        
        # Return step information
        info = {
            "net_worth": portfolio_value,
            "balance": self.balance,
            "shares_held": self.shares_held,
            "current_price": current_price
        }
        
        return next_state, reward, done, truncated, info
    
    def render(self, mode='human'):
        # Display the current state
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Shares: {self.shares_held}, "
              f"Net Worth: {self.net_worth_history[-1]:.2f}")

# --- DQN AGENT ---
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
    
    def _build_model(self):
        # Neural Net for Deep-Q learning
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        # Epsilon-greedy action selection
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        if state is None:
            return 0
        
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        # Train the model with experiences from memory
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done and next_state is not None:
                target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape(1, -1), verbose=0)[0])
            
            target_f = self.model.predict(state.reshape(1, -1), verbose=0)
            target_f[0][action] = target
            self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# --- TRAINING FUNCTION ---
def train_rl_model(data, episodes=10, batch_size=32):
    env = StockTradingEnv(data)
    state_size = 7
    action_size = 3
    agent = DQNAgent(state_size, action_size)
    
    training_history = {'episode': [], 'net_worth': [], 'reward': []}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _, info = env.step(action)
            
            if done and next_state is None:
                break
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        
        training_history['episode'].append(episode + 1)
        training_history['net_worth'].append(env.net_worth_history[-1])
        training_history['reward'].append(total_reward)
        
        progress = (episode + 1) / episodes
        progress_bar.progress(progress)
        status_text.text(f"Episode: {episode+1}/{episodes}, Final Net Worth: ${env.net_worth_history[-1]:.2f}, Total Reward: {total_reward:.4f}")
    
    status_text.empty()
    progress_bar.empty()
    
    return agent, training_history

# --- EVALUATION FUNCTION ---
def evaluate_model(agent, data):
    env = StockTradingEnv(data)
    state, _ = env.reset()
    done = False
    
    actions = []
    net_worth_history = [env.initial_balance]
    positions = ['HOLD']
    
    while not done:
        action = agent.act(state, training=False)
        next_state, reward, done, _, info = env.step(action)
        
        if done and next_state is None:
            break
        
        state = next_state
        actions.append(action)
        net_worth_history.append(info['net_worth'])
        
        if action == 0:
            positions.append('HOLD')
        elif action == 1:
            positions.append('BUY')
        else:
            positions.append('SELL')
    
    return actions, net_worth_history, positions, env.trades

@st.cache_data(ttl=3600)
def search_stock_symbol(query):
    try:
        url = f"https://financialmodelingprep.com/api/v3/search?query={query}&exchange=NSE&apikey=mxYbjV2cjX6aWCkyHZJU9QbYkwhACvSP"
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
    
    # Sidebar
    st.sidebar.title("Settings")
    stock_symbol='AAPL'  
    company_name = st.sidebar.text_input("Enter company name (e.g. Reliance Industries):")

    if company_name:
        with st.spinner("Searching for stock symbols..."):
            results = search_stock_symbol(company_name)
        
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
                    stock_symbol = st.sidebar.text_input("Enter Stock Symbol", value=selected_symbol).upper()
                    st.sidebar.success(f"Selected symbol: {selected_symbol}")
                    
                    # Now you can use selected_symbol for your analysis
            else:
                st.sidebar.warning("No matching stocks found")
      
    today = date.today()
    default_start = today - timedelta(days=365*2)
    start_date = st.sidebar.date_input("Start Date", default_start)
    end_date = st.sidebar.date_input("End Date", today)
    
    if start_date < end_date:
        with st.spinner('Loading data...'):
            data = load_data(stock_symbol, start_date, end_date)
            
        if data is not None:
            st.sidebar.success(f"Loaded {len(data)} days of data for {stock_symbol}")
            
            with st.spinner('Processing data...'):
                processed_data = preprocess_data(data)
            
            tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Technical Analysis", "Reinforcement Learning", "About"])
            
            with tab1:
                st.header(f"{stock_symbol} Stock Overview")
                
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
                st.header("Technical Analysis")
                
                st.subheader("Technical Indicators")
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
                st.header("Reinforcement Learning Stock Prediction")
                
                st.subheader("Training Parameters")
                col1, col2 = st.columns(2)
                
                with col1:
                    train_test_split = st.slider("Training/Testing Split", 0.5, 0.9, 0.8, 0.05)
                    episodes = st.slider("Training Episodes", 5, 50, 10)
                
                with col2:
                    batch_size = st.slider("Batch Size", 16, 128, 32, 16)
                    initial_balance = st.number_input("Initial Balance ($)", 1000, 100000, 10000, 1000)
                
                split_idx = int(len(processed_data) * train_test_split)
                train_data = processed_data.iloc[:split_idx]
                test_data = processed_data.iloc[split_idx:]
                
                if st.button("Train RL Model"):
                    with st.spinner("Training model... This may take a few minutes."):
                        agent, training_history = train_rl_model(train_data, episodes=episodes, batch_size=batch_size)
                    
                    st.success("Model training completed!")
                    
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                                       subplot_titles=("Net Worth During Training", "Rewards Per Episode"))
                    
                    fig.add_trace(go.Scatter(x=training_history['episode'], y=training_history['net_worth'], 
                                            mode='lines+markers', name='Net Worth'), row=1, col=1)
                    
                    fig.add_trace(go.Scatter(x=training_history['episode'], y=training_history['reward'], 
                                            mode='lines+markers', name='Total Reward'), row=2, col=1)
                    
                    fig.update_layout(height=500, title_text="Training Performance")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("Model Evaluation on Test Data")
                    
                    with st.spinner("Evaluating model on test data..."):
                        actions, net_worth_history, positions, trades = evaluate_model(agent, test_data)
                    
                    test_dates = test_data.index
                    
                    eval_df = pd.DataFrame({
                        'Date': test_dates,
                        'Close': test_data['close'].values,
                        'Action': positions[:-1],
                        'NetWorth': net_worth_history[:-1]
                    })
                    
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
                    
                    final_balance = net_worth_history[-1]
                    initial_investment = initial_balance
                    total_return = (final_balance - initial_investment) / initial_investment * 100
                    
                    st.subheader("Performance Metrics")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Final Balance", f"${final_balance:.2f}")
                    
                    with col2:
                        st.metric("Total Return", f"{total_return:.2f}%")
                    
                    with col3:
                        returns = np.diff(net_worth_history) / net_worth_history[:-1]
                        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
                        st.metric("Sharpe Ratio", f"{sharpe_ratio:.4f}")
                    
                    st.subheader("Trade Summary")
                    buy_count = positions.count('BUY')
                    sell_count = positions.count('SELL')
                    hold_count = positions.count('HOLD')
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Buy Signals", buy_count)
                    col2.metric("Sell Signals", sell_count)
                    col3.metric("Hold Signals", hold_count)
            
            with tab4:
                st.header("About This App")
                
                st.write("""
                This application uses Yahoo Finance data to visualize stock prices and technical indicators.
                It also implements a Reinforcement Learning model (Deep Q-Network) to predict optimal trading actions.
                
                **Key Features:**
                - Real-time Stock Data
                - Technical Analysis with MA, Bollinger Bands, MACD, RSI
                - Reinforcement Learning Trading Agent
                - Interactive Visualizations
                
                **Disclaimer:**
                This app is for educational purposes only and should not be considered financial advice.
                """)
        
        else:
            st.error(f"No data found for {stock_symbol} in the specified date range.")
    
    else:
        st.error("Error: End date must be after start date")

if __name__ == "__main__":
    main()

