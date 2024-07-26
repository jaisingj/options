# importing all packages
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import datetime as dt
import yfinance as yf
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.callbacks import Callback
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go


# Load tickers from tickers.csv
tickers_df = pd.read_csv('tickers.csv')
stock = tickers_df['Symbol'].tolist()

# Sidebar for stock selection and date input
st.sidebar.header('Stock Selection')
selected_stock = st.sidebar.selectbox('Select a stock', stock, index=stock.index('AAPL'))

st.sidebar.header('Date Range')
start_date = st.sidebar.date_input('Start date', dt.date(2020, 1, 1))
end_date = st.sidebar.date_input('End date', dt.date(2024, 3, 28))

# Download stock data
df = yf.download(selected_stock, start=start_date, end=end_date)


df.tail(10)

# Display the last 10 rows of the DataFrame
st.subheader('Latest Data')
st.dataframe(df.tail(10))

df= df.reset_index()
df.head()

# Describing Data
st.subheader('Data from Jan 2021 to Mar 2023')
st.write(df.describe())


# Moving Averages
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()

#Visualizations with EMA 20, 50, 100, 200 Days
ema20 = df.Close.ewm(span=20, adjust=False).mean()
ema50 = df.Close.ewm(span=50, adjust=False).mean()
ema100 = df.Close.ewm(span=100, adjust=False).mean()
ema200 = df.Close.ewm(span=200, adjust=False).mean()


# Chart with 20 & 50 Days of Exponential Moving Average
st.subheader('Closing Price vs Time Chart with 20 & 50 Days of Exponential Moving Average')
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df['Date'], y=df.Close, mode='lines', name='Close'))
fig1.add_trace(go.Scatter(x=df['Date'], y=ema20, mode='lines', name='EMA 20'))
fig1.add_trace(go.Scatter(x=df['Date'], y=ema50, mode='lines', name='EMA 50'))
fig1.update_layout(xaxis_title='Time', yaxis_title='Price')
st.plotly_chart(fig1)

st.subheader('Closing Price vs Time Chart with 100 & 200 EMA')
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df['Date'], y=df.Close, mode='lines', name='Close'))
fig2.add_trace(go.Scatter(x=df['Date'], y=ema100, mode='lines', name='EMA 100'))
fig2.add_trace(go.Scatter(x=df['Date'], y=ema200, mode='lines', name='EMA 200'))
fig2.update_layout(xaxis_title='Time', yaxis_title='Price')
st.plotly_chart(fig2)


# Splitting Data into Training and Testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

# Extracting the dates for the test data
test_dates = df['Date'][int(len(df)*0.70):].reset_index(drop=True)


# scaling the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))
data_training_array = scaler.fit_transform(data_training)

x_train = []
y_train = []

for i in range (100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i,0])
    
x_train, y_train = np.array(x_train), np.array(y_train)

# Model building

from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

model = Sequential()
model.add(LSTM(units = 50, activation= 'relu', return_sequences= True, input_shape= (x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units = 60, activation= 'relu', return_sequences= True))
model.add(Dropout(0.3))
model.add(LSTM(units = 80, activation= 'relu', return_sequences= True))
model.add(Dropout(0.4))
model.add(LSTM(units = 120, activation= 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 1))
model.summary()

model.compile(optimizer='adam', loss= 'mean_squared_error')
model.fit(x_train, y_train, epochs = 500)


# Model Testing with previous 100 days data 
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range (100,input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i,0])
    
x_test,y_test = np.array(x_test), np.array(y_test)

# loading the model
model = load_model('stock_dl_model.keras')

y_predicted = model.predict(x_test)

scale_factor = 1/0.00729727
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


# Final Plot of Original vs Predicted price trend
st.subheader('Prediction Vs Original Trend')
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=test_dates, y=y_test, mode='lines', name='Original Price'))
fig3.add_trace(go.Scatter(x=test_dates, y=y_predicted.flatten(), mode='lines', name='Predicted Price'))
fig3.update_layout(xaxis_title='Date', yaxis_title='Price', xaxis=dict(tickformat='%Y-%m-%d'))
st.plotly_chart(fig3)

# Predict the next 7 days



model.save('stock_dl_model.keras')


