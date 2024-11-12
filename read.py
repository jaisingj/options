import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
import io
import base64

# Set the page layout
st.set_page_config(layout="wide")

# Function to convert image to base64
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

# Add custom CSS for sidebar background color and table formatting
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            background-color: #CEDDF1;
        }
        thead th {
            background-color: #CEDDF1;
            color: black;
            font-weight: bold;
            text-align: center;
            padding: 10px;
            font-size: 16px;
        }
        tbody td {
            text-align: center;
            padding: 10px;
            font-size: 15px;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 0 auto;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar content
st.sidebar.markdown("<h2 style='font-size: 18px; color: navy; text-align: center; font-weight: bold;'>Options Tracker</h2>", unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Upload your trades (CSV/Excel)", type=["csv", "xlsx"])

# Function to parse and clean amount values
def parse_amount(amount):
    if pd.isna(amount):
        return 0.0
    return float(str(amount).replace('$', '').replace('(', '-').replace(')', '').replace(',', ''))

# Function to extract Type and Strike Price from Description
def parse_description(description):
    if pd.isna(description):
        return None, None
    parts = description.split()
    type_ = parts[-2] if len(parts) > 1 else None  # Second last word should be 'Call' or 'Put'
    strike_price = parse_amount(parts[-1].replace('$', '')) if len(parts) > 0 else None  # Last word is the strike price
    return type_, strike_price

if uploaded_file is not None:
    # Load the uploaded data
    data = pd.read_csv(uploaded_file)

    # Filter only BTC and STO transactions
    data = data[data['Trans Code'].isin(['BTC', 'STO'])]

    # Convert 'Activity Date' to date format only (remove timestamp)
    data['Activity Date'] = pd.to_datetime(data['Activity Date'], errors='coerce').dt.date

    # Drop rows with NaN values in essential columns
    data = data.dropna(subset=['Activity Date', 'Instrument', 'Trans Code'])

    # Sort data by 'Instrument', 'Description', and 'Activity Date'
    data = data.sort_values(by=['Instrument', 'Description', 'Activity Date', 'Trans Code'])

    # Separate BTC and STO transactions
    btc_data = data[data['Trans Code'] == 'BTC']
    sto_data = data[data['Trans Code'] == 'STO']

    # 1. Rolled Transactions: Match BTC and STO on the same date
    rolled_data = pd.merge(btc_data, sto_data, on=['Instrument', 'Activity Date'], suffixes=('_BTC', '_STO'))
    rolled_data['Net Premium'] = rolled_data['Amount_STO'].apply(parse_amount) + rolled_data['Amount_BTC'].apply(parse_amount)
    rolled_data['Activity Date_STO'] = rolled_data['Activity Date']
    rolled_data['Activity Date_BTC'] = rolled_data['Activity Date']

    # Extract Type and Strike Price from Description
    rolled_data[['Type', 'Strike Price']] = rolled_data['Description_BTC'].apply(lambda x: pd.Series(parse_description(x)))

    # Select relevant columns, placing Activity Date as the second column
    rolled_data = rolled_data[[
        'Instrument', 'Activity Date', 'Description_BTC', 'Type', 'Strike Price', 'Activity Date_STO', 'Activity Date_BTC', 
        'Amount_STO', 'Price_STO', 'Amount_BTC', 'Price_BTC', 'Net Premium'
    ]].rename(columns={'Description_BTC': 'Description'})

    # 2. Non-Roll Transactions: STO followed by BTC on a later date
    non_roll_transactions = []
    for _, sto_row in sto_data.iterrows():
        matching_btc = btc_data[
            (btc_data['Instrument'] == sto_row['Instrument']) &
            (btc_data['Description'] == sto_row['Description']) &
            (btc_data['Activity Date'] > sto_row['Activity Date'])
        ]
        if not matching_btc.empty:
            btc_row = matching_btc.iloc[0]
            net_premium = parse_amount(sto_row['Amount']) + parse_amount(btc_row['Amount'])
            type_, strike_price = parse_description(sto_row['Description'])
            non_roll_transactions.append({
                'Instrument': sto_row['Instrument'],
                'Activity Date': sto_row['Activity Date'],
                'Description': sto_row['Description'],
                'Type': type_,
                'Strike Price': strike_price,
                'Activity Date_STO': sto_row['Activity Date'],
                'Activity Date_BTC': btc_row['Activity Date'],
                'Amount_STO': sto_row['Amount'],
                'Price_STO': sto_row['Price'],
                'Amount_BTC': btc_row['Amount'],
                'Price_BTC': btc_row['Price'],
                'Net Premium': net_premium
            })

    non_roll_df = pd.DataFrame(non_roll_transactions)

    # 3. Standalone STO Transactions: STO without a corresponding BTC
    standalone_sto_data = sto_data[~sto_data['Activity Date'].isin(btc_data['Activity Date'])]
    standalone_sto_data = standalone_sto_data.rename(columns={'Amount': 'Amount_STO', 'Price': 'Price_STO'})
    standalone_sto_data = standalone_sto_data.assign(
        Amount_BTC=0.0, Price_BTC=0.0, Activity_Date_BTC=None
    )
    standalone_sto_data['Net Premium'] = standalone_sto_data['Amount_STO'].apply(parse_amount)
    standalone_sto_data['Activity Date_STO'] = standalone_sto_data['Activity Date']
    standalone_sto_data['Activity Date_BTC'] = None

    # Extract Type and Strike Price for standalone STO
    standalone_sto_data[['Type', 'Strike Price']] = standalone_sto_data['Description'].apply(lambda x: pd.Series(parse_description(x)))

    # Select relevant columns, placing Activity Date as the second column
    standalone_sto_data = standalone_sto_data[[
        'Instrument', 'Activity Date', 'Description', 'Type', 'Strike Price', 'Activity Date_STO', 'Activity Date_BTC', 
        'Amount_STO', 'Price_STO', 'Amount_BTC', 'Price_BTC', 'Net Premium'
    ]]

    # Combine all transaction types into a single DataFrame
    final_output = pd.concat([rolled_data, non_roll_df, standalone_sto_data], ignore_index=True)

    # Add a 'Rolled' column for transactions with the same 'Activity Date_STO' and 'Activity Date_BTC'
    final_output['Rolled'] = final_output.apply(
        lambda row: '✔️' if row['Activity Date_STO'] == row['Activity Date_BTC'] else '', axis=1
    )

    # Display the updated table with the 'Rolled' column
    st.subheader("Table with Rolled Column Indicating Same-Day STO and BTC")
    st.dataframe(final_output)

    # Monthly Summary Calculation
    final_output['Month'] = pd.to_datetime(final_output['Activity Date']).dt.to_period('M')
    monthly_summary = final_output.groupby('Month')['Net Premium'].sum().reset_index()
    monthly_summary.columns = ['Month', 'Net_Amount']

    # Add a total row to the monthly_summary table
    total_row = pd.DataFrame({'Month': ['Total'], 'Net_Amount': [monthly_summary['Net_Amount'].sum()]})
    monthly_summary_with_total = pd.concat([monthly_summary, total_row], ignore_index=True)

    # Display the updated monthly summary table with a total row
    st.subheader("Monthly Net Premium Summary with Total")
    st.dataframe(monthly_summary_with_total.style.format({"Net_Amount": "${:,.2f}"}))

    # Function to plot the monthly premium summary
    def plot_monthly_premium(monthly_summary):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(monthly_summary['Month'].astype(str), monthly_summary['Net_Amount'], color='skyblue', edgecolor='black')

        ax.set_title("Monthly Net Premium", fontsize=16, fontweight='bold')
        ax.set_xlabel("Month", fontsize=14, fontweight='bold')
        ax.set_ylabel("Net Premium ($)", fontsize=12, fontweight='bold')
        ax.set_ylim(0, monthly_summary['Net_Amount'].max() * 1.2)
        ax.grid(axis='y', linestyle='--', linewidth=0.5, color='grey')
        ax.set_xticklabels(monthly_summary['Month'].astype(str), rotation=45, ha='right', fontsize=10)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        st.pyplot(fig)

    # Display the bar chart
    st.subheader("Monthly Net Premium Bar Chart")
    plot_monthly_premium(monthly_summary)

else:
    st.write("Please upload a CSV file to proceed.")