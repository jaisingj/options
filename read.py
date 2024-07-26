import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, timedelta

# Set the app to wide mode
st.set_page_config(layout="wide")

# Function to initialize connection to SQLite database
def init_connection():
    conn = sqlite3.connect('mydatabase.db')
    return conn

# Function to create tables if they don't exist
def create_tables():
    conn = init_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS simulations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            stock TEXT,
            premium REAL,
            premium_100 REAL,
            net_profit REAL
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS monthly_summary (
            month TEXT PRIMARY KEY,
            net_profit REAL
        )
    ''')
    conn.commit()
    conn.close()

create_tables()

# Function to generate the list of Fridays from January 2024 until Dec 2025
def generate_fridays():
    start_date = datetime(2024, 1, 1)
    while start_date.weekday() != 4:  # 4 indicates Friday
        start_date += timedelta(days=1)
    end_date = datetime(2025, 12, 31)
    fridays = []
    while start_date <= end_date:
        fridays.append(start_date)
        start_date += timedelta(days=7)
    return fridays

# Generate the Fridays
fridays = generate_fridays()
fridays_str = [date.strftime('%Y-%m-%d') for date in fridays]

# Create a mapping of months to their respective dates
fridays_by_month = {}
for date in fridays:
    month_year = date.strftime('%B %Y')
    if month_year not in fridays_by_month:
        fridays_by_month[month_year] = []
    fridays_by_month[month_year].append(date.strftime('%Y-%m-%d'))

# Initialize session state
if 'selected_rows' not in st.session_state:
    st.session_state.selected_rows = []

if 'total_premium' not in st.session_state:
    st.session_state.total_premium = 0.0

if 'net_profit' not in st.session_state:
    st.session_state.net_profit = 0.0

if 'tax_rate' not in st.session_state:
    st.session_state.tax_rate = 0.0

if 'summary_df' not in st.session_state:
    st.session_state.summary_df = pd.DataFrame()

if 'edit_mode' not in st.session_state:
    st.session_state.edit_mode = False

# Default values for stock premiums
default_stock_premiums = {stock: 0.0 for stock in ['Apple', 'Nvidia', 'Netflix', 'Tesla', 'Amazon', 'Citi']}

# Clear input entries
if st.button("Clear Entries"):
    st.session_state.stock_premiums = default_stock_premiums.copy()

# Initialize stock premiums if not in session state
if 'stock_premiums' not in st.session_state:
    st.session_state.stock_premiums = default_stock_premiums.copy()

# Sidebar for month selection
current_month_year = datetime.now().strftime('%B %Y')
months = list(fridays_by_month.keys())
selected_month = st.sidebar.selectbox("Select Month", months, index=months.index(current_month_year))

# Sidebar for date selection within the selected month
if selected_month:
    selected_date = st.sidebar.selectbox("Select Trading Date", fridays_by_month[selected_month])

# Display the title with reduced font size
st.markdown("<h1 style='font-size:24px;'>Stock Option Simulator</h1>", unsafe_allow_html=True)

# Input for stock premiums
st.write(f"### Enter Premium Values for {selected_date}")
cols = st.columns(len(st.session_state.stock_premiums))
for i, stock in enumerate(st.session_state.stock_premiums.keys()):
    st.session_state.stock_premiums[stock] = cols[i].number_input(
        f"{stock} Premium:",
        min_value=0.0,
        step=0.01,
        value=st.session_state.stock_premiums[stock],
        key=f"{stock}_premium"
    )

# Function to insert data into the database
def insert_data(date, stock, premium, premium_100, net_profit):
    conn = init_connection()
    conn.execute('''
        INSERT INTO simulations (date, stock, premium, premium_100, net_profit)
        VALUES (?, ?, ?, ?, ?)
    ''', (date, stock, premium, premium_100, net_profit))
    conn.commit()
    conn.close()

# Function to update data in the database
def update_data(date, stock, premium, premium_100, net_profit):
    conn = init_connection()
    query = '''
        UPDATE simulations
        SET premium = ?, premium_100 = ?, net_profit = ?
        WHERE date = ? AND stock = ?
    '''
    conn.execute(query, (premium, premium_100, net_profit, date, stock))
    conn.commit()
    conn.close()

# Function to clear all tables
def clear_tables():
    conn = init_connection()
    conn.execute('DROP TABLE IF EXISTS simulations')
    conn.commit()
    conn.close()
    create_tables()

# Function to read data from the database
def get_data():
    conn = init_connection()
    df = pd.read_sql('SELECT * FROM simulations', conn)
    conn.close()
    return df

# Function to insert monthly summary into the database
def insert_monthly_summary(month, net_profit):
    conn = init_connection()
    conn.execute('''
        INSERT OR REPLACE INTO monthly_summary (month, net_profit)
        VALUES (?, ?)
    ''', (month, net_profit))
    conn.commit()
    conn.close()

# Function to load monthly summary from the database
def load_monthly_summary():
    conn = init_connection()
    df = pd.read_sql('SELECT * FROM monthly_summary', conn)
    conn.close()
    return df

# Buttons to add entry and save to database
col1, col2 = st.columns(2)
with col1:
    if st.button("Add Entry"):
        for stock, premium in st.session_state.stock_premiums.items():
            if premium > 0:  # Only add rows with non-zero premiums
                premium_100 = premium * 100
                net_profit = premium_100 * (1 - st.session_state.tax_rate / 100)
                st.session_state.selected_rows.append((selected_date, stock, premium, premium_100, net_profit))

with col2:
    if st.button("Save to Database"):
        for entry in st.session_state.selected_rows:
            insert_data(*entry)
        st.session_state.selected_rows = []
        st.success("Entries saved to database")

if st.session_state.edit_mode:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Update Entry"):
            for stock, premium in st.session_state.stock_premiums.items():
                premium_100 = premium * 100
                net_profit = premium_100 * (1 - st.session_state.tax_rate / 100)
                update_data(selected_date, stock, premium, premium_100, net_profit)
            st.session_state.edit_mode = False
            st.success("Entries updated in the database")

# Display current entries
st.write("### Current Entries")
if st.session_state.selected_rows:
    entries_df = pd.DataFrame(st.session_state.selected_rows, columns=["Date", "Stock", "Premium", "Premium x 100", "Net Profit"])
    st.write(entries_df)
else:
    st.write("No entries added yet.")

# Buttons to load data from database and clear database
col1, col2 = st.columns(2)
with col1:
    if st.button("Load Data from Database"):
        db_data = get_data()
        # Ensure net_profit is correctly calculated based on premium_100 and tax_rate
        db_data['net_profit'] = db_data['premium_100'] * (1 - st.session_state.tax_rate / 100)
        st.write(db_data)

with col2:
    if st.button("Clear Database"):
        clear_tables()
        st.success("Database cleared")

# Sum premiums and calculate net profit
st.write("### Calculate Premiums and Net Profit")
col1, col2 = st.columns(2)
with col1:
    if st.button("Sum Premiums"):
        if st.session_state.selected_rows:
            entries_df = pd.DataFrame(st.session_state.selected_rows, columns=["Date", "Stock", "Premium", "Premium x 100", "Net Profit"])
            st.session_state.total_premium = entries_df["Premium x 100"].sum()
            st.write(f"Total Premium: {st.session_state.total_premium:.2f}")
        else:
            st.write("No entries to sum.")

with col2:
    tax_rate = st.number_input("Enter Tax Rate (%)", min_value=0.0, max_value=100.0, step=0.1, key='tax_rate_input')
    st.session_state.tax_rate = tax_rate

if st.session_state.total_premium > 0:
    st.session_state.net_profit = st.session_state.total_premium * (1 - st.session_state.tax_rate / 100)
    st.write(f"Net Profit after {st.session_state.tax_rate}% tax: {st.session_state.net_profit:.2f}")

# Update net profit for all selected rows with the new tax rate
for i in range(len(st.session_state.selected_rows)):
    date, stock, premium, premium_100, _ = st.session_state.selected_rows[i]
    net_profit = premium_100 * (1 - st.session_state.tax_rate / 100)
    st.session_state.selected_rows[i] = (date, stock, premium, premium_100, net_profit)

# Generate monthly summary
st.write("### Monthly Net Profit Summary")
if st.session_state.selected_rows:
    entries_df = pd.DataFrame(st.session_state.selected_rows, columns=["Date", "Stock", "Premium", "Premium x 100", "Net Profit"])
    entries_df['Month'] = pd.to_datetime(entries_df['Date']).dt.to_period('M')
    monthly_summary = entries_df.groupby('Month')['Net Profit'].sum().reset_index()
    monthly_summary['Month'] = monthly_summary['Month'].astype(str)
    monthly_summary.columns = ['Month', 'Net Profit']
    st.session_state.summary_df = monthly_summary

if not st.session_state.summary_df.empty:
    st.write(st.session_state.summary_df)
else:
    st.write("No summary data available.")

# Function to insert monthly summary into the database
def insert_monthly_summary(month, net_profit):
    conn = init_connection()
    conn.execute('''
        INSERT OR REPLACE INTO monthly_summary (month, net_profit)
        VALUES (?, ?)
    ''', (month, net_profit))
    conn.commit()
    conn.close()

# Function to load monthly summary from the database
def load_monthly_summary():
    conn = init_connection()
    df = pd.read_sql('SELECT * FROM monthly_summary', conn)
    conn.close()
    return df

# Buttons to save monthly summary to database and load monthly summary from database
col1, col2 = st.columns(2)
with col1:
    if st.button("Save Monthly Summary to Database"):
        for index, row in st.session_state.summary_df.iterrows():
            insert_monthly_summary(row['Month'], row['Net Profit'])
        st.success("Monthly summary saved to database")

with col2:
    if st.button("Load Monthly Summary from Database"):
        monthly_summary_df = load_monthly_summary()
        st.write(monthly_summary_df)