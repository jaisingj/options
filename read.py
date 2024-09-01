import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from datetime import datetime
import streamlit as st
import yfinance as yf
from io import BytesIO, StringIO
import sqlite3

# Function to load data
def load_data(file):
    file_extension = file.name.split('.')[-1]

    if file_extension == 'csv':
        lines = file.readlines()
        lines = [line.decode('utf-8') for line in lines]
        lines = [line for line in lines if not line.startswith("The data provided is for informational purposes only.")]
        df = pd.read_csv(StringIO(''.join(lines)), parse_dates=['Activity Date', 'Settle Date'], on_bad_lines='skip')
    elif file_extension == 'xlsx':
        df = pd.read_excel(file, engine='openpyxl', parse_dates=['Activity Date', 'Settle Date'])
    else:
        st.error("Unsupported file format! Please upload a CSV or Excel file.")
        return pd.DataFrame()

    return df

def extract_expiry_date(description):
    try:
        date_str = description.split()[1]
        return datetime.strptime(date_str, '%m/%d/%Y')
    except Exception:
        return None

def extract_strike_price(description):
    try:
        price_str = description.split()[-1]
        return float(price_str.replace('$', ''))
    except Exception:
        return None

def clean_numeric_columns(df, columns):
    for col in columns:
        df[col] = df[col].replace({'\$': '', ',': ''}, regex=True)
        df[col] = df[col].replace({'\(': '-', '\)': ''}, regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def get_last_close_price(stock):
    try:
        ticker = yf.Ticker(stock)
        hist = ticker.history(period="1d")
        last_close = hist['Close'].iloc[-1]
        return last_close
    except Exception as e:
        st.warning(f"Could not retrieve closing price for {stock}: {e}")
        return np.nan

def process_data(df):
    if 'Activity Date' not in df.columns:
        st.error("The input data does not contain an 'Activity Date' column.")
        return pd.DataFrame()

    if 'Price' not in df.columns:
        st.error("The input data does not contain a 'Price' column.")
        return pd.DataFrame()

    df['Trans Code'] = df['Trans Code'].str.strip().str.upper()

    df = clean_numeric_columns(df, ['Amount', 'Price'])

    # Rename 'Instrument' column to 'Stock' and 'Price' to 'Option Price'
    df = df.rename(columns={'Instrument': 'Stock', 'Price': 'Option Price'})

    trade_df = df[df['Stock'].notnull()].copy()
    trade_df['Option Price'] = df['Option Price']
    trade_df['Activity Date'] = pd.to_datetime(trade_df['Activity Date'], errors='coerce')

    trade_df['Expiry Date'] = trade_df['Description'].apply(extract_expiry_date)
    trade_df['Strike Price'] = trade_df['Description'].apply(extract_strike_price)

    trade_df['STO Amount'] = trade_df.apply(lambda x: x['Amount'] if x['Trans Code'] == 'STO' else 0, axis=1)
    trade_df['BTC Amount'] = trade_df.apply(lambda x: x['Amount'] if x['Trans Code'] == 'BTC' else 0, axis=1)
    trade_df['STO Date'] = trade_df.apply(lambda x: x['Activity Date'] if x['Trans Code'] == 'STO' else pd.NaT, axis=1)
    trade_df['BTC Date'] = trade_df.apply(lambda x: x['Activity Date'] if x['Trans Code'] == 'BTC' else pd.NaT, axis=1)

    # Get the last close price for each stock
    trade_df['Last Close Price'] = trade_df['Stock'].apply(get_last_close_price)

    trade_df = trade_df.sort_values(by='Activity Date', ascending=False)  # Sort by most recent Activity Date

    grouped_df = trade_df.groupby(['Stock', 'Strike Price', 'Expiry Date']).agg({
        'STO Amount': 'sum',
        'BTC Amount': 'sum',
        'STO Date': 'max',
        'BTC Date': 'max',
        'Option Price': 'mean',
        'Last Close Price': 'first'  # Assuming the close price is the same for all grouped entries
    }).reset_index()

    grouped_df['Expiry Date'] = pd.to_datetime(grouped_df['Expiry Date'], errors='coerce')

    today = datetime.today()
    grouped_df['Expired'] = (grouped_df['Expiry Date'] < today) & grouped_df['BTC Date'].isna()

    grouped_df['Net Premium'] = grouped_df['STO Amount'] - grouped_df['BTC Amount'].abs()

    grouped_df = grouped_df.sort_values(by='STO Date', ascending=False)  # Sort by most recent STO Date

    grouped_df['Activity Date'] = grouped_df['STO Date'].dt.strftime('%Y-%m-%d')
    grouped_df['Expiry Date'] = grouped_df['Expiry Date'].dt.strftime('%Y-%m-%d')
    grouped_df['STO Date'] = grouped_df['STO Date'].dt.strftime('%Y-%m-%d')
    grouped_df['BTC Date'] = grouped_df['BTC Date'].dt.strftime('%Y-%m-%d')

    return grouped_df

def summarize_data(df, tax_rate):
    if 'Expiry Date' not in df.columns:
        st.error("The data does not contain an 'Expiry Date' column.")
        return pd.DataFrame()

    df['Month'] = pd.to_datetime(df['Expiry Date'], errors='coerce').dt.to_period('M')

    monthly_summary = df.groupby('Month')['Net Premium'].sum().reset_index()

    monthly_summary['Net Premium (After Tax)'] = monthly_summary['Net Premium'] * (1 - tax_rate)

    monthly_summary['Month'] = monthly_summary['Month'].dt.strftime("%b'%y")
    monthly_summary['Net Premium'] = monthly_summary['Net Premium'].astype(int)
    monthly_summary['Net Premium (After Tax)'] = monthly_summary['Net Premium (After Tax)'].astype(int)

    total_premium = monthly_summary['Net Premium'].sum()
    total_premium_after_tax = monthly_summary['Net Premium (After Tax)'].sum()

    total_row = pd.DataFrame({
        'Month': ['Total'],
        'Net Premium': [total_premium],
        'Net Premium (After Tax)': [total_premium_after_tax]
    })

    monthly_summary = pd.concat([monthly_summary, total_row], ignore_index=True)

    return monthly_summary

def plot_monthly_premium(monthly_summary):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bar_width = 0.35
    monthly_data = monthly_summary[monthly_summary['Month'] != 'Total']
    index = monthly_data.index

    bars1 = ax.bar(index - bar_width/2, monthly_data['Net Premium'], bar_width, label='Net Premium', color='steelblue', edgecolor='black', linewidth=0.6)
    bars2 = ax.bar(index + bar_width/2, monthly_data['Net Premium (After Tax)'], bar_width, label='Net Premium (After Tax)', color='green', edgecolor='black', linewidth=0.6)
    
    for rect in bars1:
        height = rect.get_height()
        ax.annotate(f'{height:,.0f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', color='black', fontsize=9)
    
    for rect in bars2:
        height = rect.get_height()
        ax.annotate(f'{height:,.0f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', color='black', fontsize=9)

    ax.set_title("Net Premium and Net Premium After Tax Per Month", fontsize=14, fontweight='bold', color='black')
    ax.set_xlabel("Month", fontsize=12, fontweight='bold', color='black')
    ax.set_ylabel("Amount ($)", fontsize=10, fontweight='bold', color='black')

    ax.tick_params(axis='both', which='both', labelsize=10, colors='black')
    ax.set_xticks(index)
    ax.set_xticklabels(monthly_data['Month'])
    ax.set_ylim(0, max(monthly_data['Net Premium'].max(), monthly_data['Net Premium (After Tax)'].max()) * 1.2)

    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(500))

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    ax.axhline(0, color='black', linewidth=0.8)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2)

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.3)
    
    st.pyplot(fig)

def highlight_total_row(s):
    return ['background-color: #E4E8F5' if s['Month'] == 'Total' else '' for _ in s]

# Function to save to database
def save_to_database(df):
    conn = sqlite3.connect('trades.db')
    df.to_sql('monthly_summary', conn, if_exists='replace', index=False)
    conn.close()

# Function to load from database
def load_from_database():
    conn = sqlite3.connect('trades.db')
    try:
        df = pd.read_sql('SELECT * FROM monthly_summary', conn)
    except Exception:
        st.error("No data available in the database.")
        df = pd.DataFrame()
    conn.close()
    return df

# Function to create an Excel file
def create_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Summary')
        
        # Access the XlsxWriter workbook and worksheet objects
        workbook  = writer.book
        worksheet = writer.sheets['Summary']
        
        # Create a chart object
        chart = workbook.add_chart({'type': 'column'})
        
        # Configure the series of the chart from the dataframe data
        chart.add_series({
            'name':       'Net Premium',
            'categories': ['Summary', 1, 0, len(df) - 2, 0],  # Skipping 'Total' row for chart
            'values':     ['Summary', 1, 1, len(df) - 2, 1],
        })
        
        chart.add_series({
            'name':       'Net Premium (After Tax)',
            'categories': ['Summary', 1, 0, len(df) - 2, 0],  # Skipping 'Total' row for chart
            'values':     ['Summary', 1, 2, len(df) - 2, 2],
        })
        
        # Add chart title and labels
        chart.set_title({'name': 'Net Premium and Net Premium After Tax Per Month'})
        chart.set_x_axis({'name': 'Month'})
        chart.set_y_axis({'name': 'Amount ($)'})
        
        # Insert the chart into the worksheet
        worksheet.insert_chart('E2', chart)  # Adjust position as needed
        
    processed_data = output.getvalue()
    return processed_data


def main():
    st.title("Options Trades Tracker")

    uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["xlsx", "csv"])

    if uploaded_file is not None:
        # If a file is uploaded, hide the Load from Database button
        data = load_data(uploaded_file)
        detailed_view = process_data(data)

        if 'Activity Date' in detailed_view.columns:
            st.subheader("Options Trades History")
            styled_detailed_view = detailed_view[['Activity Date', 'Stock', 'Last Close Price', 'Expiry Date', 'STO Date', 'BTC Date',  'Option Price', 'Strike Price',
                                                  'STO Amount', 'BTC Amount', 'Net Premium',  'Expired']].style.format("{:,.2f}", subset=['Option Price', 'Strike Price', 'STO Amount', 'BTC Amount', 'Net Premium', 'Last Close Price'])
            st.dataframe(styled_detailed_view, hide_index=True)  # Hide row numbers
        else:
            st.error("The 'Activity Date' column is missing from the processed data.")

        tax_rate = st.select_slider(
            "Select Tax Rate to Deduct from Net Premium",
            options=[0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37],
            format_func=lambda x: f"{int(x*100)}%"
        )

        monthly_summary = summarize_data(detailed_view, tax_rate)

        if not monthly_summary.empty:
            st.subheader("Monthly Net Premium Summary")
            styled_summary = monthly_summary.style.apply(highlight_total_row, axis=1).format("{:,.0f}", subset=['Net Premium', 'Net Premium (After Tax)'])
            st.dataframe(styled_summary, hide_index=True)  # Hide row numbers
            plot_monthly_premium(monthly_summary)

            # Place buttons next to each other
            col1, col2 = st.columns(2)

            with col1:
                excel_data = create_excel(monthly_summary)
                today = datetime.today().strftime('%Y-%m-%d')
                st.download_button(
                    label="Download Summary",
                    data=excel_data,
                    file_name=f'trades_{today}.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )

            with col2:
                if st.button("Save Monthly Summary"):
                    save_to_database(monthly_summary)
                    st.success("Monthly summary saved to the database!")


    else:
        # If no file is uploaded, show the Load from Database button
        if st.button("Load from Database"):
            df = load_from_database()
            if not df.empty:
                st.write("Data loaded from the database.")
                st.write(df)
                plot_monthly_premium(df)
            else:
                st.error("No data available in the database.")

if __name__ == "__main__":
    main()