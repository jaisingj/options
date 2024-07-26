import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import yfinance as yf
from datetime import datetime

# Load the tickers.csv file
tickers_df = pd.read_csv('tickers.csv')

col1, col2, col3, col4 = st.columns([0.4, 0.4, 0.2, 0.2])

with col2:
    # Create a multiselect dropdown for selecting companies
    selected_companies = st.multiselect("Select companies:", tickers_df['Name'])

# Create a dictionary to store Purchase prices and initialize with default value
Purchase_prices = {company: 0.01 for company in selected_companies}

# Create a dictionary to store sell prices and initialize with 0
sell_prices = {company: 0.01 for company in selected_companies}

# Create a dictionary to store last close prices and dates
last_close_prices = {company: {'price': 0.01, 'date': None} for company in selected_companies}

with col2:
    # Create a checkbox to automatically populate Buy Price with Last Close Price
    auto_populate_buy_price = st.checkbox("Use Last Close Price as Buy Price")

    # Create a checkbox to automatically populate Sell Price with Last Close Price
    auto_populate_sell_price = st.checkbox("Use Last Close Price as Sell Price")

# Create columns layout to put Buy and Sell input boxes side by side
col1, col2, col3, col4 = st.columns([0.3, 0.2, 0.2, 0.2])

