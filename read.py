import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
import io
import base64
import re

# Set the page layout
st.set_page_config(layout="wide")

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Add custom CSS for the sidebar background color
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            background-color: #CEDDF1; /* Sky blue color */
        }
        [data-testid="stSidebar"] .block-container {
            font-size: 14px; /* Reduce font size */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Define columns and image/text placement
col1, col2, col3, col4 = st.columns([0.4, 0.4, 0.3, 0.2])

with col1:
    image1 = Image.open('opim.jpeg')
    st.markdown(
        f"<img src='data:image/jpeg;base64,{image_to_base64(image1)}' style='max-width:70%; display:block; margin:auto; margin-top: -60px; margin-bottom: 10px;'>",
        unsafe_allow_html=True
    )

with col2:
    image1 = Image.open('opim.jpeg')
    st.markdown(
        "<h2 style='font-size: 50px; text-align: center; color: midnightblue; margin-top: -40px; margin-bottom: -40px;'>OptimuS</h2>",
    unsafe_allow_html=True
)

# Sidebar content
st.sidebar.markdown(
    "<h2 style='font-size: 18px; color: navy; text-align: center; font-weight: bold;'>Options Tracker</h2>",
    unsafe_allow_html=True
)
uploaded_file = st.sidebar.file_uploader("Upload your trades (CSV/Excel)", type=["csv", "xlsx"])


# Function to parse and clean amount values
def parse_amount(amount):
    if pd.isna(amount):
        return 0.0
    return float(str(amount).replace('$', '').replace('(', '-').replace(')', '').replace(',', ''))

# Function to extract expiration date, type, and strike price from Description
def parse_description(description):
    if pd.isna(description):
        return None, None, None
    # Match date pattern in description
    date_match = re.search(r'\d{1,2}/\d{1,2}/\d{4}', description)
    expiration_date = datetime.strptime(date_match.group(), '%m/%d/%Y').date() if date_match else None
    parts = description.split()
    type_ = parts[-2] if len(parts) > 1 else None  # Second last word should be 'Call' or 'Put'
    strike_price = parse_amount(parts[-1].replace('$', '')) if len(parts) > 0 else None  # Last word is the strike price
    return expiration_date, type_, strike_price

if uploaded_file is not None:
    # Load the uploaded data
    data = pd.read_csv(uploaded_file)

    # Filter only BTC and STO transactions
    data = data[data['Trans Code'].isin(['BTC', 'STO'])]

    # Convert 'Activity Date' to date format only (remove timestamp)
    data['Activity Date'] = pd.to_datetime(data['Activity Date'], errors='coerce').dt.date

    # Drop rows with NaN values in essential columns
    data = data.dropna(subset=['Activity Date', 'Instrument', 'Trans Code'])

    # Sort data by 'Instrument', 'Description', 'Activity Date', and 'Trans Code'
    data = data.sort_values(by=['Instrument', 'Description', 'Activity Date', 'Trans Code'])

    # Separate BTC and STO transactions
    btc_data = data[data['Trans Code'] == 'BTC']
    sto_data = data[data['Trans Code'] == 'STO']

    # 1. Rolled Transactions: Match BTC and STO on the same date
    rolled_data = pd.merge(
        btc_data,
        sto_data,
        on=['Instrument', 'Activity Date'],
        suffixes=('_BTC', '_STO')
    )
    rolled_data['Net Premium'] = (
        rolled_data['Amount_STO'].apply(parse_amount) +
        rolled_data['Amount_BTC'].apply(parse_amount)
    )
    rolled_data['Activity Date_STO'] = rolled_data['Activity Date']
    rolled_data['Activity Date_BTC'] = rolled_data['Activity Date']

    # Extract Expiration Date, Type, and Strike Price from Description
    rolled_data[['Original DTE', 'Type', 'Strike Price']] = rolled_data['Description_BTC'].apply(
        lambda x: pd.Series(parse_description(x))
    )
    rolled_data[['New DTE', 'Type_STO', 'Strike Price_STO']] = rolled_data['Description_STO'].apply(
        lambda x: pd.Series(parse_description(x))
    )

    # Select relevant columns
    rolled_data = rolled_data[
        [
            'Instrument', 'Activity Date', 'Type', 'Strike Price', 'Activity Date_STO',
            'Activity Date_BTC', 'Original DTE', 'New DTE', 'Price_STO', 'Amount_STO',
            'Price_BTC', 'Amount_BTC', 'Net Premium'
        ]
    ]

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
            net_premium = (
                parse_amount(sto_row['Amount']) +
                parse_amount(btc_row['Amount'])
            )
            original_dte, type_, strike_price = parse_description(btc_row['Description'])
            new_dte, _, _ = parse_description(sto_row['Description'])
            non_roll_transactions.append({
                #'Stock': sto_row['Instrument'],
                'Activity Date': sto_row['Activity Date'],
                #'Description_BTC': btc_row['Description'],
                #'Description_STO': sto_row['Description'],
                'Type': type_,
                'Strike Price': strike_price,
                'Activity Date_STO': sto_row['Activity Date'],
                'Activity Date_BTC': btc_row['Activity Date'],
                'Original DTE': original_dte,
                'New DTE': new_dte,
                'Amount_STO': sto_row['Amount'],
                'Price_STO': sto_row['Price'],
                'Amount_BTC': btc_row['Amount'],
                'Price_BTC': btc_row['Price'],
                'Net Premium': net_premium
            })

    non_roll_df = pd.DataFrame(non_roll_transactions)

    # 3. Standalone STO Transactions: STO without a corresponding BTC
    standalone_sto_data = sto_data[
        ~sto_data['Activity Date'].isin(btc_data['Activity Date'])
    ]
    standalone_sto_data = standalone_sto_data.rename(
        columns={'Amount': 'Amount_STO', 'Price': 'Price_STO'}
    )
    standalone_sto_data = standalone_sto_data.assign(
        #Description_BTC=None,
        Amount_BTC=0.0,
        Price_BTC=0.0,
        Activity_Date_BTC=None,
        Original_DTE=None
    )
    standalone_sto_data['Net Premium'] = standalone_sto_data['Amount_STO'].apply(parse_amount)
    standalone_sto_data['Activity Date_STO'] = standalone_sto_data['Activity Date']
    standalone_sto_data['Activity Date_BTC'] = None

    # Extract Expiration Date, Type, and Strike Price for standalone STO
    standalone_sto_data[['New DTE', 'Type', 'Strike Price']] = standalone_sto_data['Description'].apply(
        lambda x: pd.Series(parse_description(x))
    )
    standalone_sto_data['Original DTE'] = None  # No BTC date for standalone STO transactions

     # Select relevant columns
    standalone_sto_data = standalone_sto_data[
        [
            'Instrument', 'Activity Date', 'Type',
            'Strike Price', 'Activity Date_STO', 'Activity Date_BTC', 'Original DTE',
            'New DTE', 'Amount_STO', 'Price_STO', 'Amount_BTC', 'Price_BTC', 'Net Premium'
        ]
    ]


    # Combine all transaction types into a single DataFrame
    final_output = pd.concat(
        [rolled_data, non_roll_df, standalone_sto_data],
        ignore_index=True
    )

    # Add a 'Rolled' column for transactions with the same 'Activity Date_STO' and 'Activity Date_BTC'
    final_output['Rolled'] = final_output.apply(
        lambda row: '✔️' if row['Activity Date_STO'] == row['Activity Date_BTC'] else '',
        axis=1
    )

    # **Sort final_output by 'Activity Date'**
    final_output = final_output.sort_values(by='Activity Date', ascending=False)


    # Convert specified columns to numeric values, handling non-numeric entries
    for col in ["Strike Price", "Amount_STO", "Price_STO", "Amount_BTC", "Price_BTC", "Net Premium"]:
        final_output[col] = pd.to_numeric(final_output[col], errors='coerce').fillna(0.0)

    # Customize the DataFrame style for better presentation with sky blue header
    styled_final_output = final_output.style.format({
        "Strike Price": "${:,.2f}",
        "Amount_STO": "${:,.2f}",
        "Price_STO": "${:,.2f}",
        "Amount_BTC": "${:,.2f}",
        "Price_BTC": "${:,.2f}",
        "Net Premium": "${:,.2f}"
    }).set_properties(**{
        'font-size': '14px',
        'text-align': 'center'
    }).set_table_styles([
        {
            'selector': 'thead th',
            'props': [
                ('background-color', 'aliceblue'),
                ('color', 'black'),
                ('font-weight', ''),
                ('font-size', '14px'),
                ('text-align', 'center'),
                ('border', '1px solid #CCCCCC'),
                ('padding', '10px')
            ]
        },
        {
            'selector': 'tbody td',
            'props': [
                ('padding', '8px'),
                ('border', '1px solid #CCCCCC')
            ]
        },
        {
            'selector': 'table',
            'props': [
                ('width', '100%'),
                ('margin', '0 auto'),
                ('border-collapse', 'collapse')
            ]
        }
    ])

    # Display the styled DataFrame in Streamlit
    st.subheader("Transaction Details")
    st.write(styled_final_output.to_html(), unsafe_allow_html=True)

    # Monthly Summary Calculation
    final_output['Month'] = pd.to_datetime(final_output['Activity Date']).dt.to_period('M')
    monthly_summary = final_output.groupby('Month')['Net Premium'].sum().reset_index()
    monthly_summary.columns = ['Month', 'Net_Amount']

    # Add a total row to the monthly_summary table
    total_row = pd.DataFrame({
        'Month': ['Total'],
        'Net_Amount': [monthly_summary['Net_Amount'].sum()]
    })
    monthly_summary_with_total = pd.concat([monthly_summary, total_row], ignore_index=True)

    # Customize the monthly summary table
    styled_monthly_summary = monthly_summary_with_total.style.format({
        "Net_Amount": "${:,.2f}"
    }).set_properties(**{
        'font-size': '14px',
        'text-align': 'center'
    }).set_table_styles([
        {
            'selector': 'thead th',
            'props': [
                ('background-color', 'skyblue'),
                ('color', 'black'),
                ('font-weight', 'bold'),
                ('text-align', 'center'),
                ('border', '1px solid #CCCCCC'),
                ('padding', '10px')
            ]
        },
        {
            'selector': 'tbody td',
            'props': [
                ('padding', '10px'),
                ('border', '1px solid #CCCCCC')
            ]
        },
        {
            'selector': 'table',
            'props': [
                ('width', '50%'),
                ('margin', '0 auto'),
                ('border-collapse', 'collapse')
            ]
        }
    ])

    # Display the styled monthly summary table
    st.subheader("Monthly Net Premium Summary with Total")
    st.write(styled_monthly_summary.to_html(), unsafe_allow_html=True)

    # Function to plot the monthly premium summary
    def plot_monthly_premium(monthly_summary):
        # Exclude the 'Total' row from the plot
        monthly_summary = monthly_summary[monthly_summary['Month'] != 'Total']
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(
            monthly_summary['Month'].astype(str),
            monthly_summary['Net_Amount'],
            color='skyblue',
            edgecolor='black'
        )
        ax.set_title("Monthly Net Premium", fontsize=16, fontweight='bold')
        ax.set_xlabel("Month", fontsize=14, fontweight='bold')
        ax.set_ylabel("Net Premium ($)", fontsize=12, fontweight='bold')
        ax.set_ylim(0, monthly_summary['Net_Amount'].max() * 1.2)
        ax.grid(axis='y', linestyle='--', linewidth=0.5, color='grey')
        ax.set_xticks(range(len(monthly_summary['Month'])))
        ax.set_xticklabels(
            monthly_summary['Month'].astype(str),
            rotation=45,
            ha='right',
            fontsize=10
        )
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        st.pyplot(fig)

    # Display the bar chart
    st.subheader("Monthly Net Premium Bar Chart")
    plot_monthly_premium(monthly_summary_with_total)




    