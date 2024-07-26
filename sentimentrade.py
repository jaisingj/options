from imports import *
import plotly.graph_objects as go
import streamlit as st
import datetime as dt
from pandas.tseries.offsets import BDay
from app import get_news_yahoo, score_news, color_cells
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#from vaderSentiment import SentimentIntensityAnalyzer



def display_fig2(selected_ticker):


    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=180)

    # Download stock data for the selected ticker and specified date range
    data = yf.download(selected_ticker, start=start_date, end=end_date)


    # Calculate 10-day and 20-day moving averages
    data['10_MA'] = data['Close'].rolling(window=10).mean()
    data['20_MA'] = data['Close'].rolling(window=20).mean()

    # Remove the volume from the table
    data_table = data.drop('Volume', axis=1)

    # Remove the timestamp from the index
    data_table.index = data_table.index.normalize()

    # Calculate buy and sell signals
    Trade_Buy = []
    Trade_Sell = []

    # Zoomed data plot
    # Define last_180_days
    last_180_days = data_table.index[-180:]

    # Convert last_180_days to pandas DatetimeIndex
    last_180_days = pd.DatetimeIndex(last_180_days)

    # Convert lists to pandas DatetimeIndex
    Trade_Buy_Dates = pd.DatetimeIndex(Trade_Buy)
    Trade_Sell_Dates = pd.DatetimeIndex(Trade_Sell)

    # Get buy and sell signals for last 180 days
    last_180_Trade_Buy = Trade_Buy_Dates[Trade_Buy_Dates >= last_180_days[0]].tolist()
    last_180_Trade_Sell = Trade_Sell_Dates[Trade_Sell_Dates >= last_180_days[0]].tolist()

    # Create a new figure
    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(x=last_180_days, y=data_table['Close'].loc[last_180_days], mode='lines', name='Closing Price'))
    fig2.add_trace(go.Scatter(x=last_180_days, y=data_table['10_MA'].loc[last_180_days], mode='lines', name='10-day MA'))
    fig2.add_trace(go.Scatter(x=last_180_days, y=data_table['20_MA'].loc[last_180_days], mode='lines', name='20-day MA'))
    fig2.add_trace(go.Scatter(x=last_180_Trade_Buy, y=data_table.loc[last_180_Trade_Buy, 'Close'],
                              mode='markers', name='Buy Signal', marker=dict(color='green', size=8)))
    fig2.add_trace(go.Scatter(x=last_180_Trade_Sell, y=data_table.loc[last_180_Trade_Sell, 'Close'],
                              mode='markers', name='Sell Signal', marker=dict(color='red', size=8)))
    fig2.update_layout(title='Last 6-Mths Trend', xaxis_title='Date', yaxis_title='Price',
                       legend=dict(orientation="h", y=1.02, x=0.5))  # Move the legend to the bottom

    # Create a table with buy and sell signals
    data_table_with_signals = data_table.copy()
    data_table_with_signals['Signal'] = ''

    # Format the relevant columns to two decimal places
    data_table_with_signals['Open'] = data_table_with_signals['Open'].apply(lambda x: '{:.2f}'.format(x))
    data_table_with_signals['High'] = data_table_with_signals['High'].apply(lambda x: '{:.2f}'.format(x))
    data_table_with_signals['Low'] = data_table_with_signals['Low'].apply(lambda x: '{:.2f}'.format(x))
    data_table_with_signals['Close'] = data_table_with_signals['Close'].apply(lambda x: '{:.2f}'.format(x))
    data_table_with_signals['Adj Close'] = data_table_with_signals['Adj Close'].apply(lambda x: '{:.2f}'.format(x))
    data_table_with_signals['10_MA'] = data_table_with_signals['10_MA'].apply(lambda x: '{:.2f}'.format(x))
    data_table_with_signals['20_MA'] = data_table_with_signals['20_MA'].apply(lambda x: '{:.2f}'.format(x))

    # Generate 'Signal' based on 'Buy' and 'Sell' logic
    for i in range(len(data_table_with_signals) - 1):
        if ((data_table_with_signals['10_MA'].values[i] < data_table_with_signals['20_MA'].values[i]) and (data_table_with_signals['10_MA'].values[i+1] > data_table_with_signals['20_MA'].values[i+1])):
            data_table_with_signals['Signal'].iloc[i] = 'Buy'
        elif ((data_table_with_signals['10_MA'].values[i] > data_table_with_signals['20_MA'].values[i]) and (data_table_with_signals['10_MA'].values[i+1] < data_table_with_signals['20_MA'].values[i+1])):
            data_table_with_signals['Signal'].iloc[i] = 'Sell'

    # Creating columns for the layout
    #col1, col2 = st.columns([0.4, 0.2])

    # Display the full data plot in the first column
    #with col1:
    #    pass 

    # Display the stock data in a table
    # st.write(data_table_with_signals)

    news_table = get_news_yahoo(selected_ticker)
    # st.write(news_table)

    # Apply sentiment scoring to the news data
    parsed_and_scored_news = score_news(news_table)

    final_news = parsed_and_scored_news[['published', 'summary']].copy()
    final_news['published'] = pd.to_datetime(final_news['published'])
    final_news.sort_values(by='published', inplace=True)
    pd.options.display.float_format = '{:%Y-%m-%d}'.format
    # final_news

    def get_trade_open(date):
        curr_date_open = pd.to_datetime(date).floor('d').replace(hour=13, minute=30) - BDay(0)
        curr_date_close = pd.to_datetime(date).floor('d').replace(hour=20, minute=0) - BDay(0)

        prev_date_close = (curr_date_open - BDay()).replace(hour=20, minute=0)
        next_date_open = (curr_date_close + BDay()).replace(hour=13, minute=30)

        print(f"date: {date}, curr_date_open: {curr_date_open}, curr_date_close: {curr_date_close}, prev_date_close: {prev_date_close}, next_date_open: {next_date_open}")

        if ((pd.to_datetime(date) >= prev_date_close) & (pd.to_datetime(date) < curr_date_open)):
            return curr_date_open
        elif ((pd.to_datetime(date) >= curr_date_close) & (pd.to_datetime(date) < next_date_open)):
            return next_date_open
        else:
            return None

    final_news['Trading_Time'] = final_news['published'].apply(get_trade_open)
    final_news.dropna(inplace=True)
    final_news['Date'] = pd.to_datetime(pd.to_datetime(final_news['Trading_Time']).dt.date)

    vader = SentimentIntensityAnalyzer()
    scores = pd.DataFrame(final_news['summary'].apply(vader.polarity_scores).tolist())
    final_news['compound'] = scores['compound'].values.tolist()
    final_news = final_news[final_news['compound'] != 0].reset_index(drop=True)
    # st.write(final_news.head())  # display the dataframe in Streamlit

    unique_dates = final_news['Date'].unique()
    grouped_dates = final_news.groupby(['Date'])
    keys_dates = list(grouped_dates.groups.keys())

    max_score = []
    min_score = []

    for key in grouped_dates.groups.keys():
        data_group = grouped_dates.get_group(key)
        if data_group["compound"].max() > 0:
            max_score.append(data_group["compound"].max())
        elif data_group["compound"].max() < 0:
            max_score.append(0)

        if data_group["compound"].min() < 0:
            min_score.append(data_group["compound"].min())
        elif data_group["compound"].min() > 0:
            min_score.append(0)

    extreme_score = pd.DataFrame({'Date': keys_dates, 'Min_Score': min_score, 'Max_Score': max_score})
    extreme_score['Final_Score'] = extreme_score[['Min_Score', 'Max_Score']].sum(axis=1)
    
    extreme_score.head()
    # st.write(extreme_score.head())

    Buy_Option = []
    Sell_Option = []

    for i in range(len(extreme_score)):
        if extreme_score['Final_Score'].values[i] > 0.3:  # Optional Threshold
            Buy_Option.append(extreme_score['Date'].iloc[i].date())


        elif extreme_score['Final_Score'].values[i] < 0.3:  # Optional Threshold
            Sell_Option.append(extreme_score['Date'].iloc[i].date())

    vader_buy = []
    for i in range(len(data)):
        if data.index[i].date() in Buy_Option:
            vader_buy.append(i)

    vader_sell = []
    for i in range(len(data)):
        if data.index[i].date() in Sell_Option:
            vader_sell.append(i)

    # Last 30 days plot
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=data.index[-30:], y=data['Adj Close'][-30:], mode='lines', name='Closing Price'))
    fig2.add_trace(go.Scatter(x=data.index[vader_buy][-30:], y=data.loc[data.index[vader_buy][-30:], 'Adj Close'], mode='markers', name='Buy Signal', marker=dict(color='green', size=8)))
    fig2.add_trace(go.Scatter(x=data.index[vader_sell][-30:], y=data.loc[data.index[vader_sell][-30:], 'Adj Close'], mode='markers', name='Sell Signal', marker=dict(color='red', size=8)))
    fig2.update_layout(title='', xaxis_title='Date', yaxis_title='Price',
                       legend=dict(orientation="h", y=1.02, x=0.5))  # Move the legend to the bottom

    # Creating columns for the layout
    col1, col2 = st.columns([0.4, 0.2])

 
# Display the full data plot in the first column
    with col1:
         st.plotly_chart(fig2)
         st.markdown("<h5 style='color: navy; text-align: center;'>Final Signal Score (Buy: >0.3  Sell: <0.3) </h5>", unsafe_allow_html=True)


    
         # Convert the Date column to string format without the timestamp
         extreme_score['Date'] = pd.to_datetime(extreme_score['Date']).dt.strftime('%b %d')
         extreme_score[['Min_Score', 'Max_Score', 'Final_Score']] = extreme_score[['Min_Score', 'Max_Score', 'Final_Score']].applymap("{:.2f}".format)

        # Display the extreme scores table with formatted dates
         table = extreme_score.head()
         table = table.style.set_table_styles([{
             'selector': 'th',
             'props': [('background-color', '#CEDDF1'), ('color', 'black'), ('font-size', '20px')]
         }, {
             'selector': 'td',
             'props': [('font-size', '20px')]
         }])
         table = table.applymap(lambda x: f"color: {'green' if float(x) > 0 else 'red'}", subset=['Final_Score'])
    
    st.table(table)
