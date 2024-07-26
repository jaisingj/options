import pandas as pd
import plotly.graph_objs as go
import streamlit as st
import yfinance as yf
import datetime as dt
import threading  # Import threading module
import time
import requests
from io import BytesIO
from bs4 import BeautifulSoup
from datetime import datetime
import base64
import pandas_ta as ta
from plotly.subplots import make_subplots
from pandas.tseries.offsets import BDay
from app import get_news_yahoo, score_news, color_cells
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def set_sidebar_selectbox_font_size(font_size):
    st.markdown(
        f"""
        <style>
            .sidebar .widget-content .selectbox label span {{
                font-size: {font_size}px !important;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Define a helper function to get the info or display 'N/A' in red
def get_info(info, key):
    return info.get(key, "<span style='color: red;'>N/A</span>")

def display_image(img):
    img_str = image_to_base64(img)
    st.markdown(
        f'<img src="data:image/jpeg;base64,{img_str}" alt="image" style="width: 100%;">', unsafe_allow_html=True)

def display_title(main_title, subtitle):
    st.markdown(
        f"<h1 style='text-align: center;'>{main_title}</h1>", unsafe_allow_html=True)
    st.markdown(
        f"<h2 style='text-align: center;'>{subtitle}</h2>", unsafe_allow_html=True)


def display_selected_dates(start_date, end_date):
    st.markdown(f"<p>Selected Dates: <strong>{start_date}</strong> to <strong>{end_date}</strong></p>",
                unsafe_allow_html=True)


def change_progress_bar_color():
    st.markdown(
        """
        <style>
            .stProgress > div > div > div > div {
                background-color: green;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

def get_stock_industry(selected_ticker):
    return "Technology"

def create_download_link(data, filename):
    csv = data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Data</a>'
    return href

def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def get_color(value):
    return 'green' if value >= 0 else 'red'

def get_info_value(key, info):
    value = info.get(key, 'N/A') if info is not None else 'N/A'
    return f'<span style="color: red;">{value}</span>' if value == 'N/A' else value

def get_float_value(key, info):
    try:
        value = float(info.get(key, 'N/A'))
    except ValueError:  # value was 'N/A' and float('N/A') raises ValueError
        return '<span style="color: red;">N/A</span>'
    return value

def get_stock_industry(symbol):
    try:
        stock = yf.Ticker(symbol)
        stock_info = stock.info
        return stock_info.get('industry', 'N/A')
    except Exception as e:
        print(f"Error fetching industry for {symbol}: {str(e)}")
        return 'N/A'



def hint(text):
    return f"<span title='{text}'><span style='font-size: 16px; color: red; border-radius: 50%; border: 1px solid grey; padding: 0.5px 8px;'>?</span></span>"

def apply_custom_css():
    st.markdown(
        """
<style>
    table {
        width: 100%;
        text-align: center;
    }
    th {
        text-align: center;
    }

</style>
""",
        unsafe_allow_html=True,
    )

def clear_multi():
    st.session_state.symbol_multiselect = []

def color_tiers(val):
    """
    Takes a scalar and returns a string with
    the CSS property `'color: red'` or `'color: green'` depending on the value.
    """
    color = 'red' if '(' in str(val) or str(val) == 'NA' else 'green'
    return 'color: %s' % color

def hint(text):
    return f"<span title='{text}'><span style='font-size: 16px; color: red; border-radius: 50%; border: 1px solid grey; padding: 0.5px 8px;'>?</span></span>"

def apply_custom_css():
    css_path = "styles.css"
    with open(css_path, "r") as file:
        css = f"<style>{file.read()}</style>"
    return css

def get_news_yahoo(ticker):
    try:
        # Get data from Yahoo Finance
        news_data = news.get_yf_rss(ticker)
        # Convert the list of dicts into a DataFrame
        news_table = pd.DataFrame(news_data)
        return news_table
    except Exception as e:
        print(str(e))
        return pd.DataFrame()  # Return an empty DataFrame in case of an error

def color_cells(val):
    if val < 0:
        color = 'red'
    elif val > 0:
        color = 'green'
    else:
        color = 'navy'
    return 'color: %s' % color

def score_news(parsed_news_df):
    # Instantiate the sentiment intensity analyzer
    vader = SentimentIntensityAnalyzer()

    # Iterate through the headlines and get the polarity scores using vader
    scores = parsed_news_df['title'].apply(vader.polarity_scores).tolist()

    # Convert the 'scores' list of dicts into a DataFrame
    scores_df = pd.DataFrame(scores)

    # Join the DataFrames of the news and the list of dicts
    parsed_and_scored_news = parsed_news_df.join(scores_df, rsuffix='_right')
    parsed_and_scored_news = parsed_and_scored_news.rename(
        columns={"compound": "sentiment_score"})

    return parsed_and_scored_news

# Add this function to your code
def create_download_link(df, filename):
    csv_string = df.to_csv(index=False)
    b64 = base64.b64encode(csv_string.encode()).decode()
    download_link = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return download_link


def create_candlestick_chart(selected_ticker):

    # Checkbox to toggle live updates
    live_updates = st.checkbox("Live Price (Other charts will be disabled!)")

    if not live_updates:
        # Interval required 1 minute
        df = yf.download(tickers=selected_ticker, period='1d', interval='1m')
        # Calculate Exponential Moving Averages (EMA) for 20 and 50 days
        ema20 = df['Close'].ewm(span=20).mean()
        ema200 = df['Close'].ewm(span=200).mean()

        # Create checkboxes for each trace line in the chart next to each other
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            show_candlestick = st.checkbox(f'{selected_ticker}', value=True)

        with col2:
           show_ema200 = st.checkbox('200 Day', value=True)

        with col3:
           show_sp500 = st.checkbox('S&P', value=True)

        # Create a function to generate the figure based on selected checkboxes
        def generate_figure():
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # Add candlestick trace to the primary Y-axis
            if show_candlestick:
                fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'],
                                high=df['High'],
                                low=df['Low'],
                                close=df['Close'],
                                text=df['Close'],  # Text for hover tooltips (price)
                                hoverinfo='x+text',  # Show date and price on hover
                                name=f'{selected_ticker}',
                                yaxis='y', # Display on the primary Y-axis
                                increasing_fillcolor='green',  # Color for increasing candles (default is green)
                                decreasing_fillcolor='maroon',  # Set fill color to maroon for decreasing candles
                                increasing_line_width=2,  # Adjust width for increasing candles
                                decreasing_line_width=2,  # Adjust width for decreasing candles
                                decreasing_line_color='maroon'  # Set border color to maroon for decreasing candles

                ))

            # Add moving average traces to the secondary Y-axis with thinner lines
            if show_ema200:
                fig.add_trace(go.Scatter(x=df.index, y=ema200, mode='lines', name='200 Day', line=dict(color='red', dash='dot', width=2)))

            # Add S&P 500 trace to the secondary Y-axis without showing its values
            if show_sp500:
                sp500_df = yf.download('^GSPC',  period='1d', interval='1m')
                fig.add_trace(go.Scatter(x=sp500_df.index, y=sp500_df['Close'], mode='lines', name='S&P 500', line=dict(color='blue', width=1)), secondary_y=True)

            fig.update_layout(
                xaxis_title='Date',
                yaxis_title='Price',
                autosize=True,
                #width=800,
                xaxis_rangeslider_visible=True,  # Add slider for adjusting time period
                yaxis_side='left',  # Display Y-axis on the right side
                legend=dict(x=0.2, y=1.1, orientation='h'),  # Move the legend to the top of the chart and set orientation to horizontal
                height=400,  # Increase the chart height
                margin=dict(t=40),  # Add margin at the top to move the chart up
                showlegend=False if not (show_candlestick or show_ema200 or show_sp500) else True,
             #yaxis=dict(range=[0, max_y_limit]),  # Set the y-axis range
            )

            # Label for the primary Y-axis (left side)
            fig.update_yaxes(title_text='Price', showgrid=True)
            fig.update_yaxes(showgrid=False)

            # Label for the secondary Y-axis (right side)
            fig.update_yaxes(title_text='EMA', showgrid=False, secondary_y=True)
            fig.update_yaxes(showgrid=False)

            return fig

        # Generate and display the interactive candlestick chart with moving averages and S&P 500 on the secondary Y-axis
        chart_fig = generate_figure()
        st.plotly_chart(chart_fig, use_container_width=True)

    else:
        # Function to fetch and update stock data
        def fetch_stock_data(selected_ticker):
            selected_stock = yf.Ticker(selected_ticker)
            return selected_stock.history(period='1d', interval='1m')

        # Calculate the x-axis range for the entire trading day (from 9:30 AM to 4 PM)
        market_open_time = datetime.now().replace(hour=9, minute=00, second=0, microsecond=0)
        market_close_time = datetime.now().replace(hour=17, minute=0, second=0, microsecond=0)
        x_axis_range = [market_open_time, market_close_time]

        # Create an initial empty subplot with two Y-axes for the live chart
        fig_live = make_subplots(rows=1, cols=1, shared_xaxes=True, specs=[[{"secondary_y": True}]])
        fig_live.add_trace(go.Scatter(x=[], y=[], name=f'{selected_ticker} Price', line=dict(color='blue')), secondary_y=False)
        fig_live.add_trace(go.Scatter(x=[], y=[], name='S&P 500', line=dict(color='green')), secondary_y=True)

        # Configure the layout
        fig_live.update_xaxes(title_text='Time', range=x_axis_range)  # Set the x-axis range
        fig_live.update_yaxes(title_text=f'{selected_ticker} Price', secondary_y=False)
        fig_live.update_yaxes(title_text='S&P 500', secondary_y=True)


        # Create an empty container for the live chart
        chart_container_live = st.empty()

        # Function to create the live updating stock chart
        def create_live_stock_chart(selected_ticker):
            while live_updates:  # Keep updating while live updates are enabled
                ticker_history = fetch_stock_data(selected_ticker)

                # Update the Plotly chart with the latest data and x-axis range
                fig_live = make_subplots(rows=1, cols=1, shared_xaxes=True, specs=[[{"secondary_y": True}]])
                fig_live.add_trace(go.Scatter(x=ticker_history.index, y=ticker_history['Close'], mode='lines+markers', name=f'{selected_ticker} Price', line=dict(color='blue')), secondary_y=False)

                # Fetch S&P 500 data
                sp500_data = yf.download('^GSPC', period='1d', interval='1m')

                # Add the S&P 500 data to the live chart
                fig_live.add_trace(go.Scatter(x=sp500_data.index, y=sp500_data['Close'], mode='lines+markers', name='S&P 500', line=dict(color='green')), secondary_y=True)

                # Update the legend to be in the center and on top
                fig_live.update_layout(
                    width=800,
                    legend=dict(
                        orientation="h",  # Set the orientation to horizontal (top)
                        x=0.5,  # Set the legend's x-coordinate to the center
                        y=1.2,  # Set the legend's y-coordinate just above the chart
                    )
                )

                fig_live.update_xaxes(title_text='Time', range=x_axis_range)  # Set the x-axis range
                fig_live.update_yaxes(title_text=f'{selected_ticker} Price', secondary_y=False)
                fig_live.update_yaxes(title_text='S&P 500', secondary_y=True)

                # Reduce line and marker thickness
                fig_live.update_traces(
                    line=dict(width=1),  # Adjust line thickness
                    marker=dict(size=3), # Adjust marker size
                )

                fig_live.update_xaxes(showgrid=False)
                fig_live.update_yaxes(showgrid=False)

                # Update the chart in the Streamlit app
                chart_container_live.plotly_chart(fig_live)

                # Sleep for 10 seconds before fetching new data
                time.sleep(5)

        # Display the live updating chart when the checkbox is selected
        create_live_stock_chart(selected_ticker)
        if st.button("Stop"):
            st.warning("Live updates are enabled! Please turn it off before pressing stop.")
# Pass the date_range_option as an argument to generate_charts function
def generate_charts(ticker_history, start_date, end_date, selected_ticker, date_range_option):
    # Splitting the container into two columns
    ema200 = ticker_history['Close'].ewm(span=200).mean()
    col7, col8 = st.columns([0.3, 0.3])

    with col7:
        hint_text = hint("This will show you the price trend for the stock you chose and how it is moving compared to the S&P 500 index. You can use the slider bar below the chart to adjust the date range for the trend")
        st.markdown(
            f'<div class="title-container" style="margin-top: -1px; "><h2 style="color: navy; font-size: 20px;">Price Trend {hint_text}</h2></div>',
            unsafe_allow_html=True)

        # Check if the user selects '1D' in the sidebar
        if date_range_option == '1D':
            # Call the function to create the candlestick chart with S&P 500 for the selected ticker
            create_candlestick_chart(selected_ticker, )
        else:
            # Continue with the rest of your code for other date range options
            # Candlestick chart


            fig1 = go.Figure(data=[go.Candlestick(x=ticker_history.index,
                                                  open=ticker_history['Open'],
                                                  high=ticker_history['High'],
                                                  low=ticker_history['Low'],
                                                  close=ticker_history['Close'],
                                                  name='Price')])

            # Layout for Candlestick chart
            fig1.update_layout(
                autosize=True,
                #width=1200,
                #height=400,
                xaxis_title='Date',
                yaxis_title='Price',
                xaxis_rangeslider_visible=True,
                legend=dict(
                    orientation="h",
                    x=0.6,
                    y=1.3,
                    xanchor='right',
                    yanchor='top'
                ),
                margin=dict(
                    t=10,  # Adjust the top margin value as per your preference
                    l=10,
                    r=10,
                    b=10
                )
            )

            # Add 200-day Exponential Moving Average line to the chart
          
            fig1.add_trace(go.Scatter(x=ticker_history.index, y=ema200, mode='lines', name='200 Day', line=dict(color='red', dash='dot', width=2)))




            # Download S&P 500 data
            sp500_data = yf.download('^GSPC', start=start_date, end=end_date)

            # Add S&P 500 line chart to the second Y-axis
            fig1.add_trace(go.Scatter(x=sp500_data.index, y=sp500_data['Close'], mode='lines', name='S&P 500', line=dict(color='blue'), yaxis='y2'))
            # Set the second Y-axis title
            fig1.update_layout(
                yaxis2=dict(
                    title='S&P 500',
                    overlaying='y',
                    side='right',
                    position=1  # Adjust the position as per your preference
                )
            )

            fig1.update_xaxes(showgrid=False)  # Turn off x-axis gridlines
            fig1.update_yaxes(showgrid=False)  # Turn off y-axis gridlines

            # Show Candlestick chart with S&P 500 line in col7
            st.plotly_chart(fig1, use_container_width=True)





    with col8:
        # Create a title for the chart with the specified style
        hint_text = hint("This shows you the stock price with a 10 and 20 day MA. If the 10-day moving average crosses above the 20-day moving average, the price will likely fall and the stock must be sold, otherwise, it will be a signal that it is a good opportunity to buy.")
        st.markdown(
            f'<div class="title-container" style="margin-top: -2px;"><h2 style="color: navy; font-size: 20px;">Price Trend with Buy/Sell Signals {hint_text}</h2></div>',
            unsafe_allow_html=True)

        # Calculate 10-day and 20-day moving averages
        ticker_history['10_MA'] = ticker_history['Close'].rolling(
            window=10).mean()
        ticker_history['20_MA'] = ticker_history['Close'].rolling(
            window=20).mean()

        # Calculate buy and sell signals
        Trade_Buy = []
        Trade_Sell = []
        for i in range(len(ticker_history) - 1):
            if (ticker_history['10_MA'].values[i] < ticker_history['20_MA'].values[i]) and (
                    ticker_history['10_MA'].values[i + 1] > ticker_history['20_MA'].values[i + 1]):
                Trade_Buy.append(ticker_history.index[i])
            elif (ticker_history['10_MA'].values[i] > ticker_history['20_MA'].values[i]) and (
                    ticker_history['10_MA'].values[i + 1] < ticker_history['20_MA'].values[i + 1]):
                Trade_Sell.append(ticker_history.index[i])

        # Create a combined line chart with moving averages using Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ticker_history.index,
                      y=ticker_history['Close'], mode='lines', name='Closing Price'))
        fig.add_trace(go.Scatter(x=ticker_history.index,
                      y=ticker_history['10_MA'], mode='lines', name='10-day MA'))
        fig.add_trace(go.Scatter(x=ticker_history.index,
                      y=ticker_history['20_MA'], mode='lines', name='20-day MA'))
        fig.add_trace(go.Scatter(x=Trade_Buy, y=ticker_history.loc[Trade_Buy, 'Close'], mode='markers', name='Buy Signal',
                                 marker=dict(color='green', size=8)))
        fig.add_trace(go.Scatter(x=Trade_Sell, y=ticker_history.loc[Trade_Sell, 'Close'], mode='markers', name='Sell Signal',
                                 marker=dict(color='red', size=8)))

        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)


        # Calculate the price range
        price_range = ticker_history['Close'].max() - ticker_history['Close'].min()

# Determine the appropriate dtick based on the price range
        if price_range < 10:
           dtick = 1
        elif price_range < 100:
            dtick = 20
        else:
            dtick = 100

        yaxis_range = [ticker_history['Close'].min() - 10, ticker_history['Close'].max() + 10]

        
        # Title and layout
        fig.update_layout(
            title='',
            xaxis_title='Date',
            xaxis=dict(
                rangeslider=dict(
                    visible=True
                ),
             ),
            yaxis_title='Price',
            yaxis=dict(
                range=yaxis_range,
                tick0=0,  # Start at 0
                dtick=dtick,  # Major unit of 10
            ),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=1.2,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=40, r=40, t=40, b=30),
            height=400
        )

        # Show the Plotly chart
        st.plotly_chart(fig, use_container_width=True)

    col9, col10 = st.columns([0.3, 0.3])

    with col9:
        # Create a title for the chart with the specified style
        hint_text = hint("MACD is the 12-day Exponential Moving Average (EMA) minus the 26-day EMA. A 9-day EMA of MACD is plotted along side to act as a signal line.The MACD-Histogram represents the difference between MACD and  the signal line. The histogram is positive when MACD is above its 9-day EMA and negative when MACD is below its 9-day EMA.")
        st.markdown(
            f'<div class="title-container" style="margin-top: -2px;"><h2 style="color: navy; font-size: 20px;">MACD Trend {hint_text}</h2></div>',
            unsafe_allow_html=True)

        # MACD chart
        try:
            fig3 = create_macd_chart(ticker_history)
            fig3.update_layout(
                xaxis=dict(
                    title='Date',
                    title_standoff=30,  # Distance between the x-axis title and the x-axis itself
                ),
                yaxis_title='MACD',
                legend=dict(x=0.9, y=1.3, xanchor='center', yanchor='top') 
            ) 
            st.plotly_chart(fig3, use_container_width=True)
        except Exception:
            st.write(
                "MACD uses default setting for the metrics: 26/12/9 days. Refer to the FAQ for details on MACD")

    with col10:
        # Create a title for the chart with the specified style
      # Create a title for the chart with the specified style
        hint_text = hint("RSI is a momentum oscillator that measures the speed and change of price movements. The RSI oscillates between zero and 100. Traditionally the RSI is considered overbought when above 70 and oversold when below 30. Signals can be generated by looking for divergences and failure swings. RSI can also be used to identify the general trend.")
        st.markdown(
            f'<div class="title-container" style="margin-top: -2px;"><h2 style="color: navy; font-size: 20px;">RSI Trend {hint_text}</h2></div>',
            unsafe_allow_html=True)

        # RSI chart
        rsi = ta.rsi(ticker_history['Close'])
        fig4 = go.Figure(
            data=go.Scatter(x=ticker_history.index, y=rsi, name='RSI', line=dict(color='blue'), line_shape='spline'))

        # Add the horizontal line at RSI=70
        fig4.add_shape(
            type="line", line=dict(dash="dot", width=1.5, color="red"),
            y0=70, y1=70, xref='paper', x0=0, x1=1
        )
 
        fig4.update_layout(
            autosize=True,
            height=500,
            xaxis_title='Date',
            yaxis_title='RSI',
            legend=dict(
                x=1,
                y=1,
                xanchor='right',
                yanchor='top'
            )
        )


        fig4.update_xaxes(showgrid=False)  # Turn off x-axis gridlines
        fig4.update_yaxes(showgrid=False)  # Turn off y-axis gridlines

        st.plotly_chart(fig4, use_container_width=True)



def create_macd_chart(df, width=900, height=500):
    # Calculate MACD, Signal line, and Histogram using pandas_ta
    macd_df = ta.macd(df['Close'])
    macd = macd_df['MACD_12_26_9']
    signal = macd_df['MACDs_12_26_9']
    histogram = macd_df['MACDh_12_26_9']

    # Create a new DataFrame with MACD, Signal line, and Histogram
    macd_data = pd.DataFrame({
        'Date': df.index,
        'MACD': macd,
        'Signal': signal,
        'Histogram': histogram
    })

    # Drop missing or NaN values
    macd_data = macd_data.dropna()

    # Create Plotly chart for Histogram as bars
    histogram_chart = go.Bar(x=macd_data['Date'], y=macd_data['Histogram'], name='Histogram', marker=dict(color='blue'))

    # Create Plotly chart for MACD and Signal line
    macd_chart = go.Scatter(x=macd_data['Date'], y=macd_data['MACD'], name='MACD', line=dict(color='green'))
    signal_chart = go.Scatter(x=macd_data['Date'], y=macd_data['Signal'], name='Signal', line=dict(color='red'))

    # Create a subplot with two vertical subplots for MACD and Histogram
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)  # Reduce vertical spacing
    fig.add_trace(macd_chart, row=1, col=1)
    fig.add_trace(signal_chart, row=1, col=1)
    fig.add_trace(histogram_chart, row=2, col=1)

    # Update layout for the subplot
    fig.update_layout(
        autosize=True,
        width=width,
        height=height,
        xaxis_title='',
        yaxis_title='MACD',
        xaxis_rangeslider_visible=False,
        legend=dict(
            x=0.5,  # Adjust x position to move the legend to the center
            y=1,  # Adjust y position to move the legend to the top
            xanchor='center',
            yanchor='top'
        )
    )

    fig.update_xaxes(showgrid=False)  # Turn off x-axis gridlines
    fig.update_yaxes(showgrid=False)  # Turn off y-axis gridlines


    return fig


def get_stock_info(ticker):
    # Fetching the web page
    url = f"https://finance.yahoo.com/quote/{ticker}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Assuming you have a way to fetch historical_data as a DataFrame
    historical_data = fetch_historical_data(ticker)  # Placeholder for your historical data fetching logic

    # Parsing the data from the web page
    tables = soup.find_all('table')
    data = {}
    for table in tables:
        rows = table.find_all('tr')
        for row in rows:
            columns = row.find_all('td')
            if len(columns) >= 2:
                name = columns[0].text.strip()
                value = columns[1].text.strip()
                data[name] = value

    # Extracting specific information
    fifty_two_week_range = data.get('52 Week Range')
    fifty_two_week_low = fifty_two_week_range.split(' - ')[0] if fifty_two_week_range else None
    fifty_two_week_high = fifty_two_week_range.split(' - ')[1] if fifty_two_week_range else None
    current_price = "{:.2f}".format(get_last_price(ticker))  # Make sure get_last_price function is defined
    previous_close = historical_data['Close'].iloc[-2] if len(historical_data) > 1 else None


    return {
        "Beta": data.get('Beta (5Y Monthly)'),
        "Market Cap": data.get('Market Cap'),
        "P/E Ratio": data.get('PE Ratio (TTM)'),
        "Dividend Yield": data.get('Forward Dividend & Yield 4'),
        "52-Week High": fifty_two_week_high,
        "52-Week Low": fifty_two_week_low,
        "Current Price": current_price,
        "Previous Day Close": previous_close,
        "Company Description": display_company_description(ticker)  # Assuming you have this function defined elsewhere
    }


def display_stock_data(get_stock_info):
    col1, col2, col3, col4, col5 = st.columns(5)

    # Current Price with Arrow
    with col1:
        current_price = stock_info["Current Price"]      
        #price_diff = float(current_price) - float(previous_close)
        #color = "green" if price_diff > 0 else "red"
        #arrow = "&#9650;" if price_diff >= 0 else "&#9660;"
        
        st.markdown(
            f"**Current price**<br>"
            #f"{current_price} &nbsp; <span style='color:{color};'>{arrow} {'%.2f' % price_diff}</span>",
            #unsafe_allow_html=True
        )

    # Market Cap
    with col2:
        market_cap = stock_info["Market Cap"]
        st.write("Market Cap")
        st.write(market_cap)

    # PE Ratio
    with col3:
        pe_ratio = stock_info["P/E Ratio"]
        st.write("PE Ratio")
        st.write(pe_ratio)

    # 52 Week High
    with col4:
        fifty_two_week_high = stock_info["52-Week High"]
        st.write("52 Week High")
        st.write(fifty_two_week_high)

    # 52 Week Low
    with col5:
        fifty_two_week_low = stock_info["52-Week Low"]
        st.write("52 Week Low")
        st.write(fifty_two_week_low)





def display_stock_info(get_stock_info, get_float_value):
    # Splitting the container into six columns
    col1, col2, col3, col4, col5, col6 = st.columns(
        [0.4, 0.4, 0.4, 0.4, 0.4, 0.4])

    with col1:
        current_price = get_stock_info("current_Price")
        previous_close = get_info_value("previousClose", info)
        display_current_price(current_price, previous_close)

    with col2:
        market_cap = get_info_value("marketCap", info)
        display_market_cap(market_cap)

    with col3:
        pe_ratio = get_float_value("trailingPE", info)
        display_pe_ratio(pe_ratio)

    with col4:
        revenue_growth = get_info_value("revenueGrowth", info)
        display_revenue_growth(revenue_growth)

    with col5:
        fifty_two_week_high = get_info_value("fiftyTwoWeekHigh", info)
        st.markdown(f"<p class='big-label'>52 Week High</p>",
                    unsafe_allow_html=True)
        st.markdown(
            f"<p class='small-value'>{fifty_two_week_high}</p>", unsafe_allow_html=True)

    with col6:
        fifty_two_week_low = get_info_value("fiftyTwoWeekLow", info)
        st.markdown(f"<p class='big-label'>52 Week Low</p>",
                    unsafe_allow_html=True)
        st.markdown(
            f"<p class='small-value'>{fifty_two_week_low}</p>", unsafe_allow_html=True)




def display_market_cap(market_cap):
    if market_cap != '<span style="color: red;">N/A</span>':
        if isinstance(market_cap, (int, float)) and market_cap > 0:
            market_cap = float(market_cap) / 1e9  # Convert to billions
            # Format as "X.XX BN"
            market_cap = f"${'{:,.2f}'.format(market_cap)}BN"
        else:
            market_cap = 'N/A'
    st.markdown(f"<p class='big-label';background-color: #CEDDF1; color: navy>Market Cap</p>",
                unsafe_allow_html=True)
    st.markdown(
        f"<p class='small-value'>{market_cap}</p>", unsafe_allow_html=True)


def display_pe_ratio(pe_ratio):
    if pe_ratio != '<span style="color: red;">N/A</span>':
        st.markdown(f"<p class='big-label'>PE Ratio</p>",
                    unsafe_allow_html=True)
        st.markdown(
            f"<p class='small-value'>{pe_ratio:.2f}</p>", unsafe_allow_html=True)
    else:
        st.markdown(f"<p class='big-label'>PE Ratio</p>",
                    unsafe_allow_html=True)
        st.markdown(
            f"<p class='small-value'>{pe_ratio}</p>", unsafe_allow_html=True)


def display_revenue_growth(revenue_growth):
    if revenue_growth != '<span style="color: red;">N/A</span>':
        revenue_growth = f"{revenue_growth * 100:.2f}%"  # Convert to percentage
        st.markdown(f"<p class='big-label'; color: #CEDDF1; background-color: navy>Revenue Growth</p>",
                    unsafe_allow_html=True)
        st.markdown(
            f"<p class='small-value'>{revenue_growth}</p>", unsafe_allow_html=True)
    else:
        st.markdown(f"<p class='big-label'; color: #CEDDF1; background-color: navy>Revenue Growth</p>",
                    unsafe_allow_html=True)
        st.markdown(
            f"<p class='small-value'>{revenue_growth}</p>", unsafe_allow_html=True)



def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1', index=False)
    writer.close()  # This is the correct method to finalize the Excel file
    processed_data = output.getvalue()
    return processed_data

def get_table_download_link(df, text, filename):
    """Generates a link allowing the data in a given pandas dataframe to be downloaded
    in:  dataframe
    out: download link
    """
    val = to_excel(df)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64.decode()}" download="{filename}.xlsx">{text}</a>'


def plot_last_30_days_sentiment(selected_ticker, start_date):
    # Automatically set start date to 30 days ago from today
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=30)

    #data = yf.download(selected_ticker, start=start)
    
    # Retrieve and score news data
    news_table = get_news_yahoo(st.session_state.selected_ticker)
    parsed_and_scored_news = score_news(news_table)
    final_news = parsed_and_scored_news[['published', 'summary']].copy()
    final_news['published'] = pd.to_datetime(final_news['published'])
    final_news.sort_values(by='published', inplace=True)
    final_news['Trading_Time'] = final_news['published'].apply(get_trade_open)
    final_news.dropna(inplace=True)
    final_news['Date'] = pd.to_datetime(pd.to_datetime(final_news['Trading_Time']).dt.date)
    
    # Perform sentiment analysis
    vader = SentimentIntensityAnalyzer()
    scores = pd.DataFrame(final_news['summary'].apply(vader.polarity_scores).tolist())
    final_news['compound'] = scores['compound'].values.tolist()
    final_news = final_news[final_news['compound'] != 0].reset_index(drop=True)
    
    unique_dates = final_news['Date'].unique()
    grouped_dates = final_news.groupby(['Date'])
    
    max_score = []
    min_score = []
    
    for key in grouped_dates.groups.keys():
        data_group = grouped_dates.get_group(key)
        max_score.append(data_group["compound"].max())
        min_score.append(data_group["compound"].min())
        
    extreme_score = pd.DataFrame({'Date': unique_dates, 'Min_Score': min_score, 'Max_Score': max_score})
    extreme_score['Final_Score'] = extreme_score[['Min_Score','Max_Score']].sum(axis=1)
    
    Buy_Option = [d.date() for i, d in extreme_score.iterrows() if d['Final_Score'] > 0.3]
    Sell_Option = [d.date() for i, d in extreme_score.iterrows() if d['Final_Score'] < 0.3]
    
    vader_buy = [i for i in range(len(data)) if data.index[i].date() in Buy_Option]
    vader_sell = [i for i in range(len(data)) if data.index[i].date() in Sell_Option]
    
    # Create the plot for the last 30 days
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=data.index[-30:], y=data['Adj Close'][-30:], mode='lines', name='Closing Price'))
    fig2.add_trace(go.Scatter(x=data.index[vader_buy][-30:], y=data.loc[data.index[vader_buy][-30:], 'Adj Close'], mode='markers', name='Buy Signal', marker=dict(color='green', size=8)))
    fig2.add_trace(go.Scatter(x=data.index[vader_sell][-30:], y=data.loc[data.index[vader_sell][-30:], 'Adj Close'], mode='markers', name='Sell Signal', marker=dict(color='red', size=8)))
    fig2.update_layout(title='Last 30 Days Sentiment Signal', xaxis_title='Date', yaxis_title='Price',
                       legend=dict(orientation="h", y=1.02, x=0.5))
    
    st.plotly_chart(fig2)
    
def get_trade_open(date):
    curr_date_open = pd.to_datetime(date).floor('d').replace(hour=13, minute=30) - BDay(0)
    curr_date_close = pd.to_datetime(date).floor('d').replace(hour=20, minute=0) - BDay(0)
    prev_date_close = (curr_date_open - BDay()).replace(hour=20, minute=0)
    next_date_open = (curr_date_close + BDay()).replace(hour=13, minute=30)
    
    if ((pd.to_datetime(date) >= prev_date_close) & (pd.to_datetime(date) < curr_date_open)):
        return curr_date_open
    elif ((pd.to_datetime(date) >= curr_date_close) & (pd.to_datetime(date) < next_date_open)):
        return next_date_open
    else:
        return None

def get_stock_industry(symbol):
    try:
        stock = yf.Ticker(symbol)
        stock_info = stock.info
        historical_data = stock.history(period='2d')
        return stock_info.get('industry', 'N/A')
    except Exception as e:
        print(f"Error fetching industry for {symbol}: {str(e)}")
        return 'N/A'


# Function to generate a download link for a DataFrame as a CSV file
def get_table_download_link(df, text, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Convert to base64 string
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv" style="background-color: navy; color: white; padding: 8px 16px; border-radius: 4px; text-decoration: none;">{text}</a>'
    return href

def calculate_tier(percent_diff):
    if percent_diff == 0:
        return 'NA'
    elif -1 <= percent_diff < 0:
        return '0%-(1)%'
    elif -10 <= percent_diff < -1:
        return '(1%)-(10%)'
    elif -30 <= percent_diff < -10:
        return '(10%)-(30%)'
    elif -40 <= percent_diff < -30:
        return '(30%)-(40%)'
    elif -50 <= percent_diff < -40:
        return '(40%)-(50%)'
    elif percent_diff <= -50:
        return '>=(50%)'
    elif 0 < percent_diff <= 1:
        return '0%-1%'
    elif 1 < percent_diff <= 10:
        return '1%-10%'
    elif 10 < percent_diff <= 30:
        return '10%-30%'
    elif 30 < percent_diff <= 40:
        return '30%-40%'
    elif 40 < percent_diff <= 50:
        return '40%-50%'
    elif percent_diff > 50:
        return '>50%'
    else:
        return 'NA'


def simulate_future_value(symbol, target_percentage, target_dollar_amount, custom_investment=0):
    today = datetime.date.today()

    ticker_df = pd.read_csv('tickers.csv')
    names = ticker_df['Name'].tolist()

    symbol_index = names.index(symbol)
    if symbol_index >= 0:
        symbol = ticker_df['Symbol'][symbol_index]
    else:
        print(f"No symbol found for {symbol}. Skipping...")
        return None

    ticker = yf.Ticker(symbol)
    stock_info = ticker.history(period="1d")
    stock_info_p = ticker.history(period="2d")


    if not stock_info.empty:
        current_price = stock_info['Close'][0]
        target_price = current_price + (current_price * (target_percentage / 100))  # Calculate the target share price
        shares_bought = target_dollar_amount / current_price
        target_dollar_amount_after_growth = target_price * shares_bought
        gains = target_dollar_amount_after_growth - custom_investment

        gains_formatted = f'<span style="color: {"red" if gains < 0 else "green"};">({abs(gains):.2f})</span>' if gains != 0 else "0.00"


        return {
            'Company': symbol,
            'Industry': get_stock_industry(symbol),
            'Current Price': current_price,
            'Target Share Price': target_price,  # Add target share price to the result
            'Target Percentage': target_percentage,
            'Custom Investment': custom_investment,  # Include the custom investment amount
            'Target Dollar Amount': target_dollar_amount,
            'Shares': shares_bought,
            'Target Dollar Amount After Growth': target_dollar_amount_after_growth,
            'Gains': gains_formatted,  # Use the formatted gains value here
        }
    else:
        print(f"No data found for {symbol}. Skipping...")
        return None


def get_last_price(ticker):
    stock = yf.Ticker(ticker)
    todays_data = stock.history(period='1d')
    historical_data = stock.history(period='2d')
    return todays_data['Close'][0]

#def get_company_description_from_iex(ticker):
#    IEX_API_KEY = "pk_f06441175c6244478960595f048fa648"
#    url = f"https://cloud.iexapis.com/stable/stock/{ticker}/company?token={IEX_API_KEY}"
#    response = requests.get(url)
#    return response.json().get('description', 'Description not available')

#def get_company_description(ticker):
#    IEX_API_KEY = "pk_f06441175c6244478960595f048fa648"
#    url = f"https://cloud.iexapis.com/stable/stock/{ticker}/company?token={IEX_API_KEY}"
#    response = requests.get(url)
#    return response.json().get('description', 'Description not available')


def get_stock_info(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    tables = soup.find_all('table')
    
    data = {}
    for table in tables:
        rows = table.find_all('tr')
        for row in rows:
            columns = row.find_all('td')
            if len(columns) >= 2:
                name = columns[0].text.strip()
                value = columns[1].text.strip()
                data[name] = value

    fifty_two_week_range = data.get('52 Week Range')
    fifty_two_week_low = fifty_two_week_range.split(' - ')[0] if fifty_two_week_range else None
    fifty_two_week_high = fifty_two_week_range.split(' - ')[1] if fifty_two_week_range else None
    current_price = "{:.2f}".format(get_last_price(ticker))

    return {
        "Beta": data.get('Beta (5Y Monthly)'),
        "Market Cap": data.get('Market Cap'),
        "P/E Ratio": data.get('PE Ratio (TTM)'),
        "Dividend Yield": data.get('Forward Dividend & Yield 4'),
        "52-Week High": fifty_two_week_high,
        "52-Week Low": fifty_two_week_low,
        "Current Price": current_price,
        "Company Description": get_company_description_from_iex(ticker)
    }


def display_current_price(col, current_price, previous_close):
    try:
        current_price_float = float(current_price)
        previous_close_float = float(previous_close)
    except (TypeError, ValueError):
        col.markdown(
            f"<p style='color: navy;'>Current Price</p>"
            f"<p style='color:black;'>{current_price}</p>",
            unsafe_allow_html=True
        )
        col.markdown(
            f"<p style='color:red; font-size:22px; margin-top: -2px;' class='price-difference'>{previous_close}</p>",
            unsafe_allow_html=True
        )
        return

    

# Function to fetch CPI data and calculate the inflation rates
def get_inflation_data():
    headers = {'Content-type': 'application/json'}
    current_year = datetime.now().year
    data = json.dumps({
        "seriesid": ['CUUR0000SA0'],
        "startyear": str(current_year - 1),
        "endyear": str(current_year)
    })
    response = requests.post('https://api.bls.gov/publicAPI/v1/timeseries/data/', data=data, headers=headers)
    json_data = json.loads(response.text)

    # Extract CPI data
    cpi_data = json_data['Results']['series'][0]['data']

    # Get CPI for current month, previous month, and same month last year
    cpi_current_month = float(cpi_data[0]['value'])
    cpi_previous_month = float(cpi_data[1]['value'])
    cpi_same_month_last_year = float(next(item['value'] for item in cpi_data if item['year'] == str(current_year - 1)))

    # Calculate year-over-year inflation rate and month-over-month delta
    inflation_rate_yoy = ((cpi_current_month - cpi_same_month_last_year) / cpi_same_month_last_year) * 100
    delta_mom = cpi_current_month - cpi_previous_month

    return inflation_rate_yoy, delta_mom

# Function to create a speedometer chart
def create_speedometer(value, delta):
    color, arrow = ("red", "↑") if delta > 0 else ("green", "↓")
    delta_text = f"<span style='color:{color};'>{arrow}{delta:.2f}</span>"
    title_text = f"Current Inflation Rate: {value:.2f}% {delta_text}"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        gauge={'axis': {'range': [None, 10]}},
        number={'suffix': "%", 'font': {'size': 60, 'color': 'black'}},
        title={'text': title_text, 'font': {'size': 20, 'color': 'black'}}))  # Increased font size for title
    fig.update_layout(paper_bgcolor="white", width=600, height=400)
    return fig

# Streamlit app
def main():
    #st.title("Inflation Rate Tracker")

    # Get the inflation data
    inflation_rate_yoy, delta_mom = get_inflation_data()

    # Create and display the speedometer chart
    speedometer_chart = create_speedometer(inflation_rate_yoy, delta_mom)
    st.plotly_chart(speedometer_chart)

if __name__ == "__main__":
    main()



def get_stock_industry(symbol):
    try:
        stock = yf.Ticker(symbol)
        stock_info = stock.info
        return stock_info.get('industry', 'N/A')
    except Exception as e:
        print(f"Error fetching industry for {symbol}: {str(e)}")
        return 'N/A'

def get_table_download_link(df, text, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Convert to base64 string
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv" style="background-color: navy; color: white; padding: 8px 16px; border-radius: 4px; text-decoration: none;">{text}</a>'
    return href

def simulate_future_value(symbol, target_percentage, Investment, custom_investment=0):
    today = dt.date.today()

    ticker_df = pd.read_csv('tickers.csv')
    names = ticker_df['Name'].tolist()
    sec = ticker_df['Sector'].tolist()  # Use ticker_df here
    ind = ticker_df['Industry'].tolist()

    company_name = names[names.index(symbol)]
    company_sec = sec[names.index(symbol)]  # Use names.index to find the sector
    company_ind = ind[names.index(symbol)]  # Use names.index to find the industry

    symbol_index = names.index(symbol)
    if symbol_index >= 0:
        symbol = ticker_df['Symbol'][symbol_index]
    else:
        print(f"No symbol found for {symbol}. Skipping...")
        return None

    ticker = yf.Ticker(symbol)
    stock_info = ticker.history(period="1d")

    if not stock_info.empty:
        current_price = stock_info['Close'][0]
        # Convert the target_percentage to a float
        target_percentage = float(target_percentage)
        target_price = current_price * (1 + (target_percentage/100))  # Calculate the target share price
        shares_bought = Investment / current_price
        Value_after_growth = target_price * shares_bought
        gains =  Value_after_growth - Investment

        return {
            'Company': company_name,
            'Industry': company_ind,
	    'Sector':company_sec,
            'Investment': Investment,
            'Current Price': current_price,
            'Shares': shares_bought,
            'Target Percentage': target_percentage,
            'Target Share Price': target_price,  # Add target share price to the result
            'Custom Investment': custom_investment,  # Include the custom investment amount                
            'Potential Gain/Loss':  Value_after_growth,
            'Gains': gains,
        }
    else:
        print(f"No data found for {symbol}. Skipping...")
        return None


def simulate_portfolio():
    # Load the tickers.csv file
    tickers_df = pd.read_csv('tickers.csv')

    col1, col2, col3, col4 = st.columns([0.4, 0.4, 0.4, 0.4])

    with col2:
        # Create a multiselect dropdown for selecting companies
        selected_companies = st.multiselect("Select one or more Company:", tickers_df['Name'])

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
    col1, col2, col3, col4 = st.columns([0.2, 0.2, 0.2, 0.3])

    # Create input boxes for Buy and Sell prices based on user selection
    for company in selected_companies:
        with col2:
            if auto_populate_buy_price:
                # Automatically fetch the last close price and date from Yahoo Finance for Buy Price
                selected_symbol = tickers_df[tickers_df['Name'] == company]['Symbol'].values[0]
                stock_data = yf.download(selected_symbol, period="1d")
                if not stock_data.empty:
                    last_close_prices[company]['price'] = stock_data['Close'].iloc[0]
                    last_close_prices[company]['date'] = stock_data.index[0].strftime('%b %d, %y')
                    Purchase_prices[company] = last_close_prices[company]['price']
            else:
                # Allow user input for Buy price
                Purchase_prices[company] = st.number_input(f"Enter Buy Price for {company}", min_value=0.01, step=0.01, value=Purchase_prices[company])

        with col3:
            if auto_populate_sell_price:
                # Automatically fetch the last close price and date from Yahoo Finance for Sell Price
                selected_symbol = tickers_df[tickers_df['Name'] == company]['Symbol'].values[0]
                stock_data = yf.download(selected_symbol, period="1d")
                if not stock_data.empty:
                    last_close_prices[company]['price'] = stock_data['Close'].iloc[0]
                    last_close_prices[company]['date'] = stock_data.index[0].strftime('%b %d, %y')
                    sell_prices[company] = last_close_prices[company]['price']
            else:
                # Allow user input for Sell price
                sell_prices[company] = st.number_input(f"Enter Sell Price for {company}", min_value=0.01, step=0.01, value=sell_prices[company])

    col1, col2, col3, col4 = st.columns([0.2, 0.9, 0.5, 0.2])
    # Create input box for the Simulate button
    with col3:
        simulate_button = st.button("Simulate")

    # Define an empty DataFrame for results
    results_df = None

    # Check if the Simulate button is clicked
    if simulate_button:
        results = []

        for selected_company in selected_companies:
            Purchase_price = Purchase_prices[selected_company]
            sell_price = sell_prices[selected_company]
            last_close_price = last_close_prices[selected_company]['price']
            last_close_date = last_close_prices[selected_company]['date']

            formatted_results = {
                "Company Name": selected_company,
                "Industry": tickers_df[tickers_df['Name'] == selected_company]['Industry'].values[0],
                "Purchase Price($)": "{:.2f}".format(Purchase_price),
                "Sell Price($)": "{:.2f}".format(sell_price),
                "Gain/Loss($)": "{:.2f}".format(sell_price - Purchase_price),
                "Gain/Loss(%)": "{:.2f}".format(((sell_price - Purchase_price) / Purchase_price) * 100, 2)
            }

            # Include "Last Close Price" and "Last Close Date" only if auto_populate_sell_price is active
            if auto_populate_sell_price:
                formatted_results["Last Close Price"] = "{:.2f}".format(last_close_price) if last_close_price is not None else None
                formatted_results["Last Close Date"] = last_close_date if last_close_date is not None else None

            results.append(formatted_results)

        # Update the results DataFrame
        results_df = pd.DataFrame(results)

    # Determine if the simulation results in profit or loss
    if results_df is not None:
        total_buy = sum(Purchase_prices.values())
        total_gain_loss = sum(sell_prices.values()) - total_buy
        total_gain_loss_percentage = (total_gain_loss / total_buy) * 100 if total_buy > 0 else 0
        result_color = "green" if total_gain_loss >= 0 else "red"
        result_text = "Gain" if total_gain_loss >= 0 else "Loss"

    # Display the result statement and format the numbers with HTML color tags
    if results_df is not None:
        with col2:
            st.markdown(f"### Simulation Results\n\nThe current portfolio mix and projected stock prices will result in a "
                        f"<font color='{result_color}'>{result_text} of</font>  "
                        f"<font color='{result_color}'>{total_gain_loss:.2f} ({total_gain_loss_percentage:.2f}%)</font>.",
                        unsafe_allow_html=True)

    # Display the total values only if results_df is defined
    if results_df is not None:
        with col2:
            st.table(results_df)

    # Change the column title color to navy
    st.markdown(
        """
        <style>
        table th {
            color: navy !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


# Define custom CSS for the HTML table outside the function
custom_css = """
<style>
    table {width: 60%;}
    th, td {text-align: center; font-size: 14pt; min-width: 100px;}
    th {background-color: #CEDDF1;}
</style>
"""

# Call the function to simulate the portfolio
#simulate_portfolio()

