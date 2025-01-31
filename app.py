import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces

# Define the StockTradingEnv class
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
        return self._next_observation(), {}  # Return observation and info (empty dict)

    def _next_observation(self):
        # Get the current stock data and normalize it
        row = self.df.iloc[self.current_step]
        obs = np.array([
            row['Open'] / 1000,  # Normalize Open
            row['High'] / 1000,  # Normalize High
            row['Low'] / 1000,   # Normalize Low
            row['Close'] / 1000, # Normalize Close
            row['Volume'] / 1e6, # Normalize Volume
            self.balance / self.initial_balance  # Normalize Balance
        ])
        return obs

    def step(self, action):
        # Execute one step in the environment
        current_price = self.df.iloc[self.current_step]['Close']
        reward = 0

        if action == 1:  # Buy
            if self.balance >= current_price:
                self.shares_held += 1
                self.balance -= current_price
        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.shares_held -= 1
                self.balance += current_price

        # Calculate reward (change in portfolio value)
        portfolio_value = self.balance + (self.shares_held * current_price)
        reward = portfolio_value - self.initial_balance

        # Move to the next step
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        return self._next_observation(), reward, done, False, {}  # Return obs, reward, done, truncated, info

    def render(self, mode='human'):
        # Display the current state
        print(f"Step: {self.current_step}, Balance: {self.balance}, Shares Held: {self.shares_held}")

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("stocks_df.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')
    return df

df = load_data()

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Stock List", "Stock Analysis"])

# Stock List Page
if page == "Stock List":
    st.title("Top Tokens by Market Capitalization")
    
    # Calculate price change for each stock
    stock_list = df.groupby('Stock').agg({
        'Close': lambda x: x.iloc[-1] - x.iloc[-2] if len(x) > 1 else 0,
        'Volume': 'sum',
        'Close': 'last'
    }).reset_index()
    stock_list['Price Change'] = stock_list['Close']
    stock_list['Trend'] = stock_list['Price Change'].apply(lambda x: "⬆️" if x > 0 else "⬇️" if x < 0 else "➖")
    stock_list['Market Cap'] = stock_list['Close'] * 1_000_000  # Placeholder calculation

    # Display stock list with up/down arrows
    st.write("List of Stocks with Price Trends:")
    st.dataframe(stock_list[['Stock', 'Close', 'Trend', 'Price Change', 'Volume', 'Market Cap']].rename(columns={
        'Stock': 'Name',
        'Close': 'Price',
        'Trend': '24h Change',
        'Price Change': '24h Change Value',
        'Volume': '24h Volume',
        'Market Cap': 'Market Cap'
    }))

    # Add a button to select a stock
    selected_stock = st.selectbox("Select a stock for analysis", stock_list['Stock'])
    if st.button("Analyze Selected Stock"):
        st.session_state.selected_stock = selected_stock
        st.experimental_rerun()

# Stock Analysis Page
elif page == "Stock Analysis":
    st.title("Stock Analysis")

    # Check if a stock is selected
    if 'selected_stock' not in st.session_state:
        st.warning("Please select a stock from the Stock List page.")
        st.stop()

    selected_stock = st.session_state.selected_stock
    st.write(f"Analyzing: {selected_stock}")

    # Filter data for the selected stock
    stock_data = df[df['Stock'] == selected_stock]

    # Candlestick chart using Plotly
    fig = go.Figure(data=[go.Candlestick(
        x=stock_data['Date'],
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close']
    )])

    # Update layout for better visualization
    fig.update_layout(
        title=f"{selected_stock} Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=True
    )

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)

    # Reinforcement Learning Section
    st.header("Reinforcement Learning Trading Agent")

    if st.button("Train RL Agent"):
        env = StockTradingEnv(stock_data)
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=10000)
        st.success("RL Agent Trained Successfully!")

    if st.button("Evaluate RL Agent"):
        env = StockTradingEnv(stock_data)
        model = PPO.load("stock_trading_ppo")
        obs, _ = env.reset()
        done = False
        rewards = []

        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            rewards.append(reward)

        st.write(f"Total Reward: {sum(rewards)}")
        st.line_chart(rewards)
