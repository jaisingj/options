import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
import io
import base64
import re
from datetime import datetime



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
        f"<img src='data:image/jpeg;base64,{image_to_base64(image1)}' style='max-width:60%; display:block; margin:auto; margin-top: -100px; margin-bottom: 10px;'>",
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
    
    # Match various common date formats in the description
    date_match = re.search(r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', description)
    
    if date_match:
        date_str = date_match.group(1)
        try:
            # Attempt to parse the date with multiple formats
            expiration_date = datetime.strptime(date_str, '%m/%d/%Y').date()
        except ValueError:
            try:
                expiration_date = datetime.strptime(date_str, '%m-%d-%Y').date()
            except ValueError:
                try:
                    expiration_date = datetime.strptime(date_str, '%d/%m/%Y').date()
                except ValueError:
                    expiration_date = None  # Couldn't parse date
    else:
        expiration_date = None

    # Extract the option type and strike price if available
    parts = description.split()
    type_ = parts[-2] if len(parts) > 1 else None
    strike_price = parse_amount(parts[-1].replace('$', '')) if len(parts) > 0 else None

    return expiration_date, type_, strike_price

if uploaded_file is not None:
    # Load the data, skipping problematic lines
    data = pd.read_csv(uploaded_file, on_bad_lines="skip")

    # Remove any unwanted header or footer rows containing specific text
    unwanted_text = (
        "The data provided is for informational purposes only. Please consult a professional tax service or "
        "personal tax advisor if you need instructions on how to calculate cost basis or questions regarding "
        "your specific tax situation. Reminder: This data does not include Robinhood Crypto or Robinhood Spending activity."
    )
    data = data[~data.apply(lambda row: row.astype(str).str.contains(unwanted_text, regex=False).any(), axis=1)]

    # Rename 'Instrument' to 'Stock' if it exists
    if 'Instrument' in data.columns:
        data = data.rename(columns={'Instrument': 'Stock'})



    data['Price'] = data['Price'].apply(parse_amount)
    data['Amount'] = data['Amount'].apply(parse_amount)

    # Filter only BTC and STO transactions
    data = data[data['Trans Code'].isin(['BTC', 'STO'])]

    # Convert 'Activity Date' to date format only (remove timestamp)
    data['Activity Date'] = pd.to_datetime(data['Activity Date'], errors='coerce').dt.date

    # Drop rows with NaN values in essential columns
    data = data.dropna(subset=['Activity Date', 'Stock', 'Trans Code'])

    # Sort data by 'Stock', 'Description', 'Activity Date', and 'Trans Code'
    data = data.sort_values(by=['Stock', 'Description', 'Activity Date', 'Trans Code'])

    # Separate BTC and STO transactions
    btc_data = data[data['Trans Code'] == 'BTC'].dropna(subset=['Stock'])
    sto_data = data[data['Trans Code'] == 'STO'].dropna(subset=['Stock'])

    # 1. Rolled Transactions: Match BTC and STO on the same date
    rolled_data = pd.merge(
        btc_data,
        sto_data,
        on=['Stock', 'Activity Date'],
        suffixes=('_BTC', '_STO')
    )
    rolled_data['Net Premium'] = (
        rolled_data['Amount_STO'].apply(parse_amount) +
        rolled_data['Amount_BTC'].apply(parse_amount)
    )
    rolled_data['STO_Date'] = rolled_data['Activity Date']
    rolled_data['BTC_Date'] = rolled_data['Activity Date']

    # Rename columns after merge for consistency
    rolled_data = rolled_data.rename(columns={'Amount_STO': 'AMT_STO', 'Amount_BTC': 'AMT_BTC'})

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
            'Stock', 'Activity Date', 'Type', 'Strike Price', 'STO_Date',
            'BTC_Date', 'Original DTE', 'New DTE', 'Price_STO', 'AMT_STO',
            'Price_BTC', 'AMT_BTC', 'Net Premium'
        ]
    ]

    # 2. Non-Roll Transactions: STO followed by BTC on a later date
    non_roll_transactions = []
    for _, sto_row in sto_data.iterrows():
        matching_btc = btc_data[
            (btc_data['Stock'] == sto_row['Stock']) &
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
                'Stock': sto_row['Stock'],
                'Activity Date': sto_row['Activity Date'],
                'Type': type_,
                'Strike Price': strike_price,
                'STO_Date': sto_row['Activity Date'],
                'BTC_Date': btc_row['Activity Date'],
                'Original DTE': original_dte,
                'New DTE': new_dte,
                'AMT_STO': sto_row['Amount'],
                'Price_STO': sto_row['Price'],
                'AMT_BTC': btc_row['Amount'],
                'Price_BTC': btc_row['Price'],
                'Net Premium': net_premium
            })

    non_roll_df = pd.DataFrame(non_roll_transactions).dropna(subset=['Stock'])

    # For standalone STO transactions, populate Original DTE if BTC is missing
    standalone_sto_data = sto_data[~sto_data['Activity Date'].isin(btc_data['Activity Date'])]
    standalone_sto_data = standalone_sto_data.rename(columns={'Amount': 'AMT_STO', 'Price': 'Price_STO'})
    standalone_sto_data = standalone_sto_data.assign(
        AMT_BTC=0.0,
        Price_BTC=0.0,
        BTC_Date=None,
        STO_Date=standalone_sto_data['Activity Date']  # Ensure STO_Date is the same as Activity Date
    )

    # Extract expiration date, type, and strike price for standalone STO
    # Here, set Original DTE to expiration date from Description if there's no BTC match
    standalone_sto_data[['Expiration DTE', 'Type', 'Strike Price']] = standalone_sto_data['Description'].apply(
        lambda x: pd.Series(parse_description(x))
    )
    standalone_sto_data['Original DTE'] = standalone_sto_data['Expiration DTE']  # Use Expiration DTE as Original DTE

    # Drop NaN in Stock for standalone STO data
    standalone_sto_data = standalone_sto_data.dropna(subset=['Stock'])
    
    # Combine all transaction types into a single DataFrame
    final_output = pd.concat([rolled_data, non_roll_df, standalone_sto_data], ignore_index=True)

    # Ensure proper date conversion and handle missing values for 'New DTE' and 'BTC_Date'
    final_output['Original DTE'] = pd.to_datetime(final_output['Original DTE'], errors='coerce').dt.date
    final_output['New DTE'] = pd.to_datetime(final_output['New DTE'], errors='coerce').dt.date
    final_output['New DTE'] = final_output['New DTE'].fillna('')  # Replace missing values with blank strings
    final_output['BTC_Date'] = pd.to_datetime(final_output['BTC_Date'], errors='coerce').dt.date
    final_output['BTC_Date'] = final_output['BTC_Date'].fillna('')  # Replace missing values with blank strings

    # Add a 'Status' column with all conditions
    final_output['Status'] = final_output.apply(
        lambda row: 'Rolled' if row['STO_Date'] == row['BTC_Date'] else
                    ('Closed' if row['BTC_Date'] and row['Original DTE'] == row['New DTE'] else
                    ('Expired' if row['BTC_Date'] and row['New DTE'] == '' and row['Original DTE'] and 
                      row['Original DTE'] < datetime.now().date() else
                    ('Expired' if not row['BTC_Date'] and row['Original DTE'] and 
                      row['Original DTE'] < datetime.now().date() else
                    ('Open' if not row['BTC_Date'] and row['New DTE'] == '' else '')))),
        axis=1
    )

    # Adjust 'Net Premium' based on the status
    final_output['Net Premium'] = final_output.apply(
        lambda row: row['AMT_STO'] if row['Status'] == 'Open' else
                    (row['AMT_STO'] + row['AMT_BTC'] if row['Status'] in ['Rolled', 'Expired'] else row['Net Premium']),
        axis=1
    )

    # Ensure no missing values are carried forward in 'Net Premium'
    final_output['Net Premium'] = final_output['Net Premium'].fillna(0)
            # Sort final_output by 'Activity Date'

    final_output = final_output.sort_values(by='Activity Date', ascending=False)

    # Convert specified columns to numeric values, handling non-numeric entries
    for col in ["Strike Price", "AMT_STO", "Price_STO", "AMT_BTC", "Price_BTC", "Net Premium"]:
        final_output[col] = pd.to_numeric(final_output[col], errors='coerce').fillna(0.0)

    # Select relevant columns up to "Net Premium" for display
    final_output_display = final_output[
        [
            'Stock', 'Activity Date', 'Type', 'Strike Price', 'STO_Date',
            'BTC_Date', 'Original DTE', 'New DTE', 'Price_STO', 'AMT_STO',
            'Price_BTC', 'AMT_BTC', 'Net Premium', 'Status'
        ]
    ]

    # Customize the DataFrame style for better presentation with sky blue header
    styled_final_output = final_output_display.style.format({
        "Strike Price": "${:,.2f}",
        "AMT_STO": "${:,.2f}",
        "Price_STO": "${:,.2f}",
        "AMT_BTC": "${:,.2f}",
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

    # Add custom CSS for button styling
    st.markdown("""
        <style>
        .stDownloadButton > button {
            background-color: #0A57C1 !important; /* Filled green background */
            color: white !important; /* White text */
            border: none !important;
            padding: 10px 20px !important;
            border-radius: 8px !important;
            font-size: 16px !important;
            font-weight: bold !important;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2); /* Shadow for depth */
            transition: background-color 0.3s ease; /* Smooth transition */
            margin-top: 10px !important; /* Adds space above the button */
        }
        .stDownloadButton > button:hover {
            background-color: #388E3C !important; /* Darker green on hover */
        }
    </style>
    """, unsafe_allow_html=True)


    st.download_button(
        label="Download Transaction CSV",
        data=final_output.to_csv(index=False).encode("utf-8"),
        file_name="transaction_details.csv",
        mime="text/csv",
        key="transaction_csv"
    )



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

    # Tax rate selection and "Net After Tax" calculation
    col1, col2, col3 = st.columns([0.4, 0.4, 0.3])
    with col1:
        st.subheader("Tax Bracket")
        tax_rate = st.select_slider(
            "Select Tax Rate to Deduct from Net Premium",
            options=[0, 0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37],
            format_func=lambda x: f"{int(x*100)}%"
        )

        monthly_summary_with_total['Net After Tax'] = monthly_summary_with_total['Net_Amount'] * (1 - tax_rate)

        styled_monthly_summary = monthly_summary_with_total.style.format({
            "Net_Amount": "${:,.2f}",
            "Net After Tax": "${:,.2f}"
        }).set_properties(**{
            'font-size': '14px',
            'text-align': 'center'
        }).set_table_styles([
            {
                'selector': 'thead th',
                 'props': [
                    ('background-color', 'aliceblue'),
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
                ('width', '100%'),
                ('margin', '0 auto'),
                ('border-collapse', 'collapse')
            ]
        }
    ])


    col1, col2, col3 = st.columns([0.4, 0.5, 0.1])

    with col1:
        st.subheader("Monthly Summary")
        st.write(styled_monthly_summary.to_html(), unsafe_allow_html=True)


        # Convert monthly_summary_with_total DataFrame to CSV
        summary_csv = monthly_summary_with_total.to_csv(index=False)
        st.download_button(
            label="Download SummaryCSV",
            data=summary_csv,
            file_name="monthly_summary.csv",
            mime="text/csv"
        )


    # Plotting function
    def plot_monthly_premium(monthly_summary):
        monthly_summary = monthly_summary[monthly_summary['Month'] != 'Total']
        monthly_summary['Month'] = monthly_summary['Month'].astype(str)

        fig, ax = plt.subplots(figsize=(10, 6))
        width = 0.35
        x = range(len(monthly_summary['Month']))

        ax.bar(
            [i - width / 2 for i in x],
            monthly_summary['Net_Amount'],
            width=width,
            label="Net Premium",
            color='navy',
            edgecolor='black'
        )

        ax.bar(
            [i + width / 2 for i in x],
            monthly_summary['Net After Tax'],
            width=width,
            label="Net After Tax",
            color='green',
            edgecolor='black',
            alpha=0.7
        )

        #ax.set_title("Monthly Net Premium", fontsize=16, fontweight='bold')
        ax.set_xlabel("Month", fontsize=14, fontweight='bold')
        ax.set_ylabel("Net Premium ($)", fontsize=12, fontweight='bold')
    
        max_value = max(monthly_summary['Net_Amount'].max(), monthly_summary['Net After Tax'].max())
        ax.set_ylim(0, max_value * 1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(monthly_summary['Month'], rotation=45, ha='right')

        # Set light grey dotted gridlines
        #ax.grid(axis='y', color='lightgrey', linestyle=':', linewidth=0.8)

        # Increase font size for Y-axis values
        ax.tick_params(axis='y', labelsize=12)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.legend()

        st.pyplot(fig)

    with col2:
        st.subheader("Monthly Net Premium")
        plot_monthly_premium(monthly_summary_with_total)
