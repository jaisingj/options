import streamlit as st
import pandas as pd
from datetime import datetime, date
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import re
import io
import base64
from PIL import Image

# ----------------------------------------------------------------------------
# Streamlit App Setup
# ----------------------------------------------------------------------------
st.set_page_config(layout="wide")

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Custom CSS for the sidebar
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            background-color: #C2E0FF;
        }
        [data-testid="stSidebar"] .block-container {
            font-size: 14px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
def parse_amount(amount):
    """
    Cleans monetary strings like '$1,234.56', '(1,234.56)', or merged strings.
    Uses regex to extract multiple numeric pieces and sum them if found.
    """
    if pd.isna(amount):
        return 0.0
    text = str(amount)
    # Remove $ and commas, convert parentheses to negative sign
    text = text.replace('$', '').replace(',', '')
    text = text.replace('(', '-').replace(')', '')
    # Use regex to grab all signed floats
    matches = re.findall(r'[+-]?\d+(?:\.\d+)?', text)
    if not matches:
        return 0.0
    # Sum all numeric pieces
    total = sum(float(m) for m in matches)
    return total

def extract_expiry_date(description):
    """Extract an option expiration date (e.g., '1/19/2024') from text."""
    if pd.isna(description):
        return None
    match = re.search(r'\b(\d{1,2}/\d{1,2}/\d{4})\b', str(description))
    if match:
        return datetime.strptime(match.group(1), '%m/%d/%Y').date()
    return None

def tag_transactions(merged_data):
    """
    Tag each row (Expired, Closed, Rolled, Open, or Unknown).
    Adjust logic as needed for your definitions.
    """
    def assign_status(row):
        current_date = date.today()

        # We'll do a quick example check using a derived "btc_price" from BTC Amount
        btc_price = abs(row['BTC Amount'] / 100) if row['BTC Amount'] != 0 else 0

        # Expired: No BTC, Expiry < current_date
        if pd.isna(row['BTC Date']) and pd.notna(row['Expiry Date']) and row['Expiry Date'] < current_date:
            return "Expired"

        # Closed: Has BTC Date, and BTC Price < 2
        elif pd.notna(row['BTC Date']) and (btc_price < 2):
            return "Closed"

        # Rolled: Has BTC & STO both, big BTC Amount
        elif (
            pd.notna(row['BTC Date']) 
            and pd.notna(row['STO Date']) 
            and abs(row['BTC Amount']) > 2
        ):
            return "Rolled"

        # Open: Has STO, no BTC, future expiry
        elif (
            pd.notna(row['STO Date']) 
            and pd.isna(row['BTC Date']) 
            and row['Expiry Date'] > current_date
        ):
            return "Open"

        return "Unknown"

    merged_data['Status'] = merged_data.apply(assign_status, axis=1)
    return merged_data

def compute_quantity(row):
    """
    If STO Amount != 0, use (STO Amount / (STO Price * 100)).
    Else if BTC Amount != 0, use (BTC Amount / (BTC Price * 100)).
    Otherwise, return 0.
    """
    if row['STO Amount'] != 0:
        amt = row['STO Amount']
        price = row['STO Price']
    elif row['BTC Amount'] != 0:
        amt = row['BTC Amount']
        price = row['BTC Price']
    else:
        return 0

    if pd.isna(price) or price == 0:
        return 0
    return amt / (price * 100)

# ----------------------------------------------------------------------------
# Display header images and title
# ----------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns([0.6, 0.5, 0.5, 0.5])
with col2:
    image1 = Image.open('opin.jpeg')
    st.markdown(
        f"""
        <div style="text-align: center; margin-top: -60px;">
            <img src='data:image/jpeg;base64,{image_to_base64(image1)}' style='max-width:80%; display:block; margin:auto;'>
        </div>
        """,
        unsafe_allow_html=True
    )

# ----------------------------------------------------------------------------
# File Upload + Data Preprocessing
# ----------------------------------------------------------------------------
uploaded_file = st.sidebar.file_uploader("Upload your trades (CSV/Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # 1) Load data
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file, on_bad_lines="skip")
    else:
        data = pd.read_excel(uploaded_file)

    # 2) Strip whitespace from column names
    data.columns = data.columns.str.strip()

    # 3) Parse numeric fields to floats
    if 'Amount' in data.columns:
        data['Amount'] = data['Amount'].apply(parse_amount)
    if 'Price' in data.columns:
        data['Price'] = data['Price'].apply(parse_amount)
    # If 'Quantity' exists, parse it too (optional)
    if 'Quantity' in data.columns:
        data['Quantity'] = data['Quantity'].apply(parse_amount)

    # Debug: show raw data
    #st.write("### Raw file columns:", data.columns.tolist())
    #st.write("### Raw data preview:")
    #st.write(data.head(10))

    # -------------------------------------------------------------------
    # Consolidate partial fills: group by identifying columns so a single
    # 2-contract trade won't appear as two 1-contract lines, etc.
    # -------------------------------------------------------------------
    unique_cols = [
        "Instrument",
        "Trans Code",
        "Activity Date",
        "Description",
        # Add 'Process Date' or 'Settle Date' here if needed
    ]
    agg_dict = {}
    if 'Amount' in data.columns:
        agg_dict['Amount'] = 'sum'
    if 'Price' in data.columns:
        # e.g., average fill price for partials
        agg_dict['Price'] = 'mean'

    data_consolidated = (
        data
        .groupby(unique_cols, as_index=False, dropna=False)
        .agg(agg_dict)
    )

    #st.write("### Consolidated Partial Fills Preview")
    #st.write(data_consolidated.head(10))

    # 4) Ensure essential columns are not missing
    essential_cols = ['Activity Date', 'Instrument', 'Trans Code']
    existing_essentials = [col for col in essential_cols if col in data_consolidated.columns]
    data_consolidated = data_consolidated.dropna(subset=existing_essentials)

    # 5) Extract Expiry Date from Description
    if 'Description' in data_consolidated.columns:
        data_consolidated['Expiry Date'] = data_consolidated['Description'].apply(extract_expiry_date)

    # -------------------------------------------------------------------
    # Separate STO vs BTC
    # -------------------------------------------------------------------
    sto_data = data_consolidated[data_consolidated['Trans Code'] == 'STO']
    btc_data = data_consolidated[data_consolidated['Trans Code'] == 'BTC']

    # Group STO
    sto_agg = {
        "Amount": "sum",
        "Price": "mean",
        "Activity Date": "min",
    }
    sto_grouped = (
        sto_data
        .groupby(["Description", "Instrument", "Expiry Date"], dropna=False)
        .agg(sto_agg)
        .reset_index()
        .rename(columns={"Amount": "STO Amount", "Price": "STO Price", "Activity Date": "STO Date"})
    )

    # Group BTC
    btc_agg = {
        "Amount": "sum",
        "Price": "mean",
        "Activity Date": "max",
    }
    btc_grouped = (
        btc_data
        .groupby(["Description", "Instrument"], dropna=False)
        .agg(btc_agg)
        .reset_index()
        .rename(columns={"Amount": "BTC Amount", "Price": "BTC Price", "Activity Date": "BTC Date"})
    )

    # Merge STO & BTC
    merged_data = pd.merge(
        sto_grouped,
        btc_grouped,
        on=["Description", "Instrument"],
        how="outer"
    )

    # Fill missing numeric columns with 0
    merged_data['STO Amount'] = merged_data['STO Amount'].fillna(0)
    merged_data['BTC Amount'] = merged_data['BTC Amount'].fillna(0)
    merged_data['STO Price'] = merged_data['STO Price'].fillna(0)
    merged_data['BTC Price'] = merged_data['BTC Price'].fillna(0)

    # Combine the Activity Date
    merged_data['Activity Date'] = merged_data[['STO Date', 'BTC Date']].apply(
        lambda row: min([d for d in row if pd.notna(d)]) if any(pd.notna(d) for d in row) else None,
        axis=1
    )

    # Net Premium (Amount = STO + BTC)
    merged_data['Amount'] = merged_data['STO Amount'] + merged_data['BTC Amount']

    # Convert Activity Date to monthly period
    merged_data['Activity Month'] = pd.to_datetime(merged_data['Activity Date'], errors='coerce').dt.to_period('M')

    # -----------------------------------------------------------------------
    # Compute Quantity from STO vs. BTC
    # -----------------------------------------------------------------------
    merged_data['Quantity'] = merged_data.apply(compute_quantity, axis=1)
    merged_data['Quantity'] = merged_data['Quantity'].round(0).astype(int)

    # Tag transactions
    merged_data = tag_transactions(merged_data)

    # Debug example
    #st.write("### Merged Data (e.g. AMZN if any):")
    #st.write(merged_data[merged_data['Instrument'] == 'AMZN'])

    # -----------------------------------------------------------------------
    # Monthly Summary
    # -----------------------------------------------------------------------
    monthly_summary = (
        merged_data
        .groupby('Activity Month')['Amount']
        .sum()
        .reset_index()
        .rename(columns={'Amount': 'Net Premium'})
        .sort_values(by='Activity Month')
    )

    # Add a grand total
    grand_total = monthly_summary['Net Premium'].sum()
    monthly_summary = pd.concat(
        [
            monthly_summary,
            pd.DataFrame({'Activity Month': ['Grand Total'], 'Net Premium': [grand_total]})
        ],
        ignore_index=True
    )

    # Tax Rate & Net After Tax
    colA, colB, colC = st.columns([0.2, 0.4, 0.2])
    with colB:
        st.subheader("Tax Bracket")
        tax_rate = st.select_slider(
            "Select Tax Rate to Deduct from Net Premium",
            options=[0, 0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37],
            format_func=lambda x: f"{int(x * 100)}%"
        )

    monthly_summary['Net After Tax'] = monthly_summary['Net Premium'] * (1 - tax_rate)
    grand_total_net_after_tax = monthly_summary[monthly_summary['Activity Month'] != 'Grand Total']['Net After Tax'].sum()
    monthly_summary.loc[monthly_summary['Activity Month'] == 'Grand Total', 'Net After Tax'] = grand_total_net_after_tax

    # -----------------------------------------------------------------------
    # Styled Monthly Summary Table
    # -----------------------------------------------------------------------
    def highlight_grand_total(row):
        if row['Activity Month'] == 'Grand Total':
            return ['background-color: aliceblue; color: black; font-weight: bold; text-align: center'] * len(row)
        else:
            return [''] * len(row)

    styled_monthly_summary = (
        monthly_summary
        .style
        .format({"Net Premium": "${:,.2f}", "Net After Tax": "${:,.2f}"})
        .set_properties(**{'font-size': '18px', 'text-align': 'center'})
        .set_table_styles([
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
        .apply(highlight_grand_total, axis=1)
    )

    col1, col2, col3 = st.columns([0.3, 0.4, 0.1])
    with col1:
        st.subheader("Summary")
        st.write(styled_monthly_summary.to_html(), unsafe_allow_html=True)
        summary_csv = monthly_summary.to_csv(index=False)
        st.download_button(
            label="Download Summary CSV",
            data=summary_csv,
            file_name="monthly_summary.csv",
            mime="text/csv"
        )

    # -----------------------------------------------------------------------
    # Bar Chart for Monthly Net Premium
    # -----------------------------------------------------------------------
    def plot_monthly_premium(monthly_summary):
        chart_data = monthly_summary[monthly_summary['Activity Month'] != 'Grand Total']
        chart_data['Activity Month'] = chart_data['Activity Month'].astype(str)

        fig, ax = plt.subplots(figsize=(16, 12))
        width = 0.35
        x = range(len(chart_data['Activity Month']))

        ax.bar(
            [i - width / 2 for i in x],
            chart_data['Net Premium'],
            width=width,
            label="Net Premium",
            color='navy',
            edgecolor='black'
        )
        ax.bar(
            [i + width / 2 for i in x],
            chart_data['Net After Tax'],
            width=width,
            label="Net After Tax",
            color='green',
            edgecolor='black',
            alpha=0.7
        )

        ax.set_xlabel("Month", fontsize=18, fontweight='bold')
        ax.set_ylabel("Net Premium ($)", fontsize=14, fontweight='bold')

        min_value = min(chart_data['Net Premium'].min(), chart_data['Net After Tax'].min())
        max_value = max(chart_data['Net Premium'].max(), chart_data['Net After Tax'].max())
        ax.set_ylim(min_value * 1.2 if min_value < 0 else 0, max_value * 1.2 if max_value > 0 else 0)

        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(chart_data['Activity Month'], rotation=45, ha='right')

        ax.tick_params(axis='y', labelsize=16)
        ax.tick_params(axis='x', labelsize=20)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.legend(fontsize=18, loc='upper left')
        st.pyplot(fig)

    with col2:
        st.subheader("Monthly Net Premium")
        plot_monthly_premium(monthly_summary)

    # -----------------------------------------------------------------------
    # Pie Charts (Open Positions & Positive Premium Distribution)
    # -----------------------------------------------------------------------
    colA, colB, colC = st.columns([0.3, 0.5, 0.5])
    with colB:
        selected_month = st.selectbox(
            "Select Month",
            options=monthly_summary[monthly_summary['Activity Month'] != 'Grand Total']['Activity Month']
                .astype(str).unique(),
            index=0,
            key="month_filter_pie_col1"
        )
        filtered_data = merged_data[
            pd.to_datetime(merged_data['Activity Month'].astype(str)) == pd.to_datetime(selected_month)
        ]

    pie_col1, pie_col2 = st.columns([0.5, 0.5])

    # 1) Pie Chart: Open Positions by (computed) Quantity
    with pie_col1:
        open_positions = filtered_data[filtered_data['Status'] == 'Open']
        total_open_qty = open_positions['Quantity'].sum()

        if not open_positions.empty and total_open_qty > 0:
            stock_distribution = (
                open_positions
                .groupby('Instrument')['Quantity']
                .sum()
                .reset_index()
            )
            stock_distribution['Percentage'] = (stock_distribution['Quantity'] / total_open_qty) * 100

            def autopct_format(pct, all_vals):
                absolute = int(round(pct / 100.0 * sum(all_vals)))
                return f"{pct:.1f}%\n({absolute})"

            fancy_colors = [
                "#4E79A7", "#1f77b4", "#59A14F", "#E15759", "#36A2EB",
                "#0A57C1", "#1367D8", "#FF9D9A", "#1367D8", "#BAB0AC"
            ]
            glow_colors = [mcolors.to_rgba(c, alpha=0.3) for c in fancy_colors[:len(stock_distribution)]]

            fig1, ax1 = plt.subplots(figsize=(2, 2))
            # Background glow pie
            ax1.pie(
                stock_distribution['Percentage'],
                labels=None,
                startangle=140,
                radius=1.05,
                colors=glow_colors,
                wedgeprops={'linewidth': 0}
            )
            # Main pie
            wedges, texts, autotexts = ax1.pie(
                stock_distribution['Percentage'],
                labels=stock_distribution['Instrument'],
                autopct=lambda pct: autopct_format(pct, stock_distribution['Quantity']),
                startangle=140,
                colors=fancy_colors[:len(stock_distribution)],
                textprops={'color': 'black', 'fontsize': 4}
            )
            plt.setp(autotexts, color='white', size=4)
            plt.title(f"Open Positions ({selected_month})", fontsize=8)
            st.pyplot(fig1)
        else:
            st.write("No open positions for the selected month.")

    # 2) Pie Chart: Positive Premium (Closed/Expired)
    with pie_col2:
        positive_premium_positions = filtered_data[
            (filtered_data['Status'].isin(['Closed', 'Expired'])) & (filtered_data['Amount'] > 0)
        ]
        stock_positive_premium = (
            positive_premium_positions
            .groupby('Instrument')['Amount']
            .sum()
            .reset_index()
        )

        if not stock_positive_premium.empty:
            fig2, ax2 = plt.subplots(figsize=(2, 2))
            wedges, texts, autotexts = ax2.pie(
                stock_positive_premium['Amount'],
                labels=stock_positive_premium['Instrument'],
                autopct='%1.1f%%',
                startangle=140,
                colors=plt.cm.Paired.colors,
                textprops={'color': 'black', 'fontsize': 4}
            )
            plt.setp(autotexts, color='white', size=4)
            plt.title(f"Closed/Expired Premium ({selected_month})", fontsize=6)
            plt.setp(autotexts, size=8)
            st.pyplot(fig2)
        else:
            st.write("No closed/expired transactions with positive premium for the selected month.")

    # -----------------------------------------------------------------------
    # Detailed Transactions - SORT DESCENDING BY ACTIVITY DATE
    # -----------------------------------------------------------------------
    # Make 'Activity Month' first column if present
    if 'Activity Month' in merged_data.columns:
        first_col = ['Activity Month']
        other_cols = [c for c in merged_data.columns if c not in first_col]
        merged_data = merged_data[first_col + other_cols]

    # 1) Convert 'Activity Date' to datetime to ensure proper sorting
    merged_data['Activity Date'] = pd.to_datetime(merged_data['Activity Date'], errors='coerce')

    # 2) Sort descending by 'Activity Date'
    sorted_transactions = merged_data.sort_values(by='Activity Date', ascending=False)

    # Convert date columns to datetime so .strftime won't error if some are strings
    date_cols = ['STO Date', 'BTC Date']
    for dc in date_cols:
        if dc in sorted_transactions.columns:
            sorted_transactions[dc] = pd.to_datetime(sorted_transactions[dc], errors='coerce')

    # 3) Style and display
    styled_transactions = (
        sorted_transactions
        .style
        .format({
            "STO Amount": "${:,.2f}",
            "BTC Amount": "${:,.2f}",
            "Amount": "${:,.2f}",
            "STO Price": "${:,.2f}",
            "BTC Price": "${:,.2f}",
            "Activity Date": lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else '',
            "STO Date": lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else '',
            "BTC Date": lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else ''
        })
        .set_properties(**{'font-size': '15px', 'text-align': 'center'})
        .set_table_styles([
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
    )

    st.write("### Detailed Transactions")
    st.write(styled_transactions.to_html(), unsafe_allow_html=True)

    # Custom CSS for Download Button
    st.markdown(
        """
        <style>
        .stDownloadButton > button {
            background-color: #0A57C1 !important;
            color: white !important;
            border: none !important;
            padding: 10px 20px !important;
            border-radius: 8px !important;
            font-size: 16px !important;
            font-weight: bold !important;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
            transition: background-color 0.3s ease;
            margin-top: 10px !important;
        }
        .stDownloadButton > button:hover {
            background-color: #388E3C !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Download button for detailed transactions
    detailed_csv = sorted_transactions.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Detailed Transactions",
        data=detailed_csv,
        file_name="detailed_transactions.csv",
        mime="text/csv"
    )