import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Function to get the previous non-weekend day
def get_previous_non_weekend_day(date):
    while True:
        date -= timedelta(days=1)
        if date.weekday() < 5:
            return date

# Function to calculate the percentage difference and format it
def calculate_percentage_difference(current_price, reference_price):
    if reference_price == 0:
        return "N/A"
    percentage_difference = ((current_price - reference_price) / reference_price) * 100
    formatted_percentage = f"{percentage_difference:.2f}%"
    return formatted_percentage

# Function to categorize percentage difference into 10 groups
def categorize_percentage_difference(percentage_difference):
    if percentage_difference >= -100 and percentage_difference < -80:
        return "-100% to -80%"
    elif percentage_difference >= -80 and percentage_difference < -60:
        return "-80% to -60%"
    elif percentage_difference >= -60 and percentage_difference < -40:
        return "-60% to -40%"
    elif percentage_difference >= -40 and percentage_difference < -20:
        return "-40% to -20%"
    elif percentage_difference >= -20 and percentage_difference < 0:
        return "-20% to 0%"
    elif percentage_difference >= 0 and percentage_difference < 20:
        return "0% to 20%"
    elif percentage_difference >= 20 and percentage_difference < 40:
        return "20% to 40%"
    elif percentage_difference >= 40 and percentage_difference < 60:
        return "40% to 60%"
    elif percentage_difference >= 60 and percentage_difference < 80:
        return "60% to 80%"
    elif percentage_difference >= 80 and percentage_difference <= 100:
        return "80% to 100%"
    else:
        return "N/A"

# Function to get ROE, P/E ratio, and Debt-to-Equity ratio
def get_company_info(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get("returnOnEquity"), info.get("trailingPE"), info.get("debtToEquity")
    except:
        return None, None, None

# Set the app layout to wide mode
st.set_page_config(layout="wide")

# Streamlit UI
st.title("Stock Info Analyzer")

# Create a new DataFrame to store results
result_data = []

# Create columns for layout
col1, col2, col3 = st.columns([1, 2, 1])

# Upload CSV file in col1
uploaded_file = col2.file_uploader("Upload a CSV file with company names and tickers", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV
    df = pd.read_csv(uploaded_file)
    
    # Get the total number of stocks
    num_stocks = len(df)
    
    # Initialize progress bar in col2
    progress_bar = col2.progress(0)
    
    # Initialize timer text in col2
    timer_text = col2.empty()
    
    # Start time
    start_time = datetime.now()
    
    # Loop through each row in the CSV
    for index, row in df.iterrows():
        company_name = row["Name"]
        ticker = row["Symbol"]
        
        try:
            data = yf.download(ticker)
            
            if not data.empty:
                # Filter data for the first half of 2020 (January 1 to June 30)
                data_2020_first_half = data.loc["2020-01-01":"2020-06-30"]
                
                if not data_2020_first_half.empty:
                    # Find the date and value of the highest price in the first half of 2020
                    max_price_date = data_2020_first_half["High"].idxmax()
                    max_price_value = data_2020_first_half.loc[max_price_date]["High"]
                    
                    # Get the latest closing price and date
                    last_closing_date = get_previous_non_weekend_day(datetime.now()).strftime("%Y-%m-%d")
                    last_closing_price = data["Close"].loc[last_closing_date]
                    
                    # Calculate the percentage difference between last closing price and highest price
                    percentage_difference = calculate_percentage_difference(last_closing_price, max_price_value)
                    
                    # Categorize the percentage difference into groups
                    percentage_group = categorize_percentage_difference(float(percentage_difference.strip('%')))
                    
                    # Get ROE, P/E ratio, and Debt-to-Equity ratio
                    roe, pe_ratio, debt_equity_ratio = get_company_info(ticker)
                    
                    # Append the data to the result list
                    result_data.append([company_name, max_price_value, max_price_date.strftime("%Y-%m-%d"), last_closing_price, last_closing_date, percentage_difference, percentage_group, roe, pe_ratio, debt_equity_ratio])
            
            # Update progress bar
            processed_symbols = index + 1
            progress = processed_symbols / num_stocks
            progress_bar.progress(progress)
            
            # Estimate remaining time in minutes
            remaining_time_minutes = estimate_remaining_time(start_time, num_stocks, processed_symbols)
            timer_text.text(f"Estimated Remaining Time: {remaining_time_minutes} minutes")
        except Exception as e:
            pass  # Skip stocks where data cannot be found
    
    if result_data:
        # Create a DataFrame from the result list
        result_df = pd.DataFrame(result_data, columns=["Company Name", "Highest Price in H1 2020", "Date of Highest Price", "Last Closing Price", "Last Closing Date", "Percentage Difference", "Percentage Group", "ROE", "P/E Ratio", "Debt to Equity Ratio"])
        
        # Display the result table in col3
        col2.write("Result Table:")
        col2.write(result_df)
