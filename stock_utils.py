import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
from millify import millify
import requests
import time

def get_last_price(ticker):
    stock = yf.Ticker(ticker)
    todays_data = stock.history(period='1d')
    if not todays_data.empty:
        return todays_data['Close'].iloc[0]
    return None

def display_company_description(ticker):
    selected_ticker = st.session_state.get('selected_ticker')
    if selected_ticker:
        attempts = 10
        for attempt in range(attempts):
            try:
                company = yf.Ticker(selected_ticker)
                business_summary = company.info.get('longBusinessSummary', 'No summary available.')
                st.markdown(f"<style>.big-font {{font-size: 21px;}}</style>", unsafe_allow_html=True)
                st.markdown(f'<div class="big-font">Business Summary for {selected_ticker}: {business_summary}</div>', unsafe_allow_html=True)
                return
            except requests.exceptions.HTTPError as e:
                #st.warning(f"")
                time.sleep(5)
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}.")
                return
        st.error("App usage levels are high, Yahoo Finance API limit may impact results. Please try again after a while.")
    else:
        st.write('No company selected.')

def format_market_cap(value):
    if value is None:
        return "N/A"
    else:
        return millify(value, precision=2)

def format_pe_ratio(value):
    if value is None:
        return "N/A"
    else:
        return f"{value:.2f}"

def display_error_message():
    st.error("Application usage level is high, loading times may be longer than usual...")

def get_stock_metrics(ticker):
    attempts = 10
    for attempt in range(attempts):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            market_cap = info.get('marketCap')
            formatted_market_cap = format_market_cap(market_cap)
            pe_ratio = info.get('trailingPE')
            formatted_pe_ratio = format_pe_ratio(pe_ratio)

            metrics = {
                "Beta (5Y Monthly)": info.get('beta'),
                "Market Cap": formatted_market_cap,
                "P/E Ratio (TTM)": formatted_pe_ratio,
                "Forward Dividend & Yield": info.get('dividendYield'),
                "52-Week High": info.get('fiftyTwoWeekHigh'),
                "52-Week Low": info.get('fiftyTwoWeekLow'),
                "Current Price": info.get('currentPrice'),
                "Previous Day Close": info.get('previousClose')
            }
            
            return metrics
        except requests.exceptions.HTTPError as e:
            #st.warning(f"HTTPError encountered: {e}. Retrying ({attempt + 1}/{attempts})...")
            time.sleep(5)
        except Exception as e:
            #st.error(f"An unexpected error occurred: {e}.")
            return None
    display_error_message()
    return None

def display_current_price(col, current_price, previous_close):
    try:
        current_price_float = float(current_price)
        previous_close_float = float(previous_close)
    except (TypeError, ValueError):
        col.markdown(
            f"<p style='color: navy; font-size:21px;'>Current Price</p>"
            f"<p style='color:black;'>{current_price}</p>",
            unsafe_allow_html=True
        )
        col.markdown(
            f"<p style='color:red; font-size:21px; margin-top: -2px;' class='price-difference'>{previous_close}</p>",
            unsafe_allow_html=True
        )
        return

    # Calculate price difference
    price_difference = current_price_float - previous_close_float
    price_diff_percent = (price_difference / previous_close_float) * 100
    color = "green" if price_difference >= 0 else "red"
    arrow = "&#9650;" if price_difference >= 0 else "&#9660;"

    col.markdown(
        f"<p style='color: black; font-size:21px;'>Current Price</p>"
        f"<p style='color: navy; font-size:21px;'>{current_price} &nbsp; <span style='color:{color};'>{arrow} {'%.2f' % price_difference} ({'%.2f' % price_diff_percent}%)</span></p>",
        unsafe_allow_html=True
    )

def display_stock_info():
    # Check for selected ticker in session state or default to 'AAPL'
    ticker = st.session_state.get('selected_ticker', 'AAPL')
    stock_info = get_stock_metrics(ticker)

    if stock_info is None:
        return

    current_price = stock_info.get("Current Price")
    fifty_two_week_low = stock_info.get("52-Week Low")
    
    # Check if the required data is available
    if not current_price or not fifty_two_week_low:
        st.markdown("<span style='color: red; font-size: 20px;'>Key Metrics N/A.</span>", unsafe_allow_html=True)
        return

    # Layout
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    display_current_price(col1, stock_info["Current Price"], stock_info["Previous Day Close"])

    with col2:
        st.markdown("<p style='color: black; font-size:21px;'>Market Cap</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: navy; font-size:21px;'>{stock_info['Market Cap']}</p>", unsafe_allow_html=True)

    with col3:
        st.markdown("<p style='color: black; font-size:21px;'>P/E Ratio</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: navy; font-size:21px;'>{stock_info['P/E Ratio (TTM)']}</p>", unsafe_allow_html=True)

    with col4:
        st.markdown("<p style='color: black; font-size:21px;'>Dividend Yield</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: navy;font-size:21px;'>{stock_info['Forward Dividend & Yield']}</p>", unsafe_allow_html=True)

    with col5:
        st.markdown("<p style='color: black; font-size:21px;'>52 Week High</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: navy;font-size:21px;'>{stock_info['52-Week High']}</p>", unsafe_allow_html=True)

    with col6:
        st.markdown("<p style='color: black; font-size:21px;'>52 Week Low</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: navy;font-size:21px;'>{stock_info['52-Week Low']}</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    st.title("Stock Metrics Viewer")
    st.text_input("Enter Stock Ticker Symbol", "AAPL", key='selected_ticker')
    if st.button("Get Metrics"):
        display_stock_info()
