import streamlit as st
import pandas as pd
from datetime import datetime, date
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import re
import io
import base64
from PIL import Image

# Streamlit App Setup
st.set_page_config(layout="wide")

# Helper functions
# Helper functions
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Add custom CSS for the sidebar
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #C2E0FF;
        }
        [data-testid="stSidebar"] .block-container {
            font-size: 14px;
        }
    </style>
    """, unsafe_allow_html=True)


# Helper Functions
def parse_amount(amount):
    """Clean monetary values."""
    if pd.isna(amount):
        return 0.0
    return float(str(amount).replace('$', '').replace('(', '-').replace(')', '').replace(',', ''))

def extract_expiry_date(description):
    """Extract expiration date from the description."""
    if pd.isna(description):
        return None
    date_match = re.search(r'\b(\d{1,2}/\d{1,2}/\d{4})\b', description)
    if date_match:
        return datetime.strptime(date_match.group(1), '%m/%d/%Y').date()
    return None

def tag_transactions(merged_data):
    """Tag transactions based on the updated conditions."""
    def assign_status(row):
        current_date = date.today()

        # Derive BTC Price from BTC Amount
        btc_price = abs(row['BTC Amount'] / 100) if row['BTC Amount'] != 0 else 0

        # Expired: No BTC Date/Amount and Expiry Date < current date
        if pd.isna(row['BTC Date']) and pd.notna(row['Expiry Date']) and row['Expiry Date'] < current_date:
            return "Expired"

        # Closed: BTC Date exists, BTC Price < 2
        elif pd.notna(row['BTC Date']) and btc_price < 2:
            return "Closed"

        # Rolled: BTC Date exists, followed by an STO Date (same Description), and abs(BTC Amount) > 2
        elif (
            pd.notna(row['BTC Date']) 
            and pd.notna(row['STO Date']) 
            #and row['BTC Date'] < row['STO Date'] 
            and abs(row['BTC Amount']) > 2
        ):
            return "Rolled"

        # Open: STO exists, no BTC, and Expiry Date > current date
        elif pd.notna(row['STO Date']) and pd.isna(row['BTC Date']) and row['Expiry Date'] > current_date:
            return "Open"

        # Default: Return "Unknown" for unhandled cases
        return "Unknown"

    # Apply tagging logic to each row
    merged_data['Status'] = merged_data.apply(assign_status, axis=1)
    return merged_data

# Display header images and title
col1, col2, col3, col4 = st.columns([0.6, 0.5, 0.5, 0.5])

with col1:
    image1 = Image.open('bck1.jpg')
    st.markdown(
        f"""
        <div style="opacity: 0.2; text-align: center; margin-top: -110px;">
            <img src='data:image/jpeg;base64,{image_to_base64(image1)}' style='max-width:105%; display:block; margin:auto;'>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    image1 = Image.open('opin.jpeg')
    st.markdown(
        f"""
        <div style="text-align: center; margin-top: -100px;">
            <img src='data:image/jpeg;base64,{image_to_base64(image1)}' style='max-width:120%; display:block; margin-auto:'>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    image1 = Image.open('bck1.jpg')
    st.markdown(
        f"""
        <div style="opacity: 0.3; text-align: center; margin-top: -104px;">
            <img src='data:image/jpeg;base64,{image_to_base64(image1)}' style='max-width:105%; display:block; margin:auto;'>
        </div>
        """,
        unsafe_allow_html=True
    )


with col4:
    image1 = Image.open('bck1.jpg')
    st.markdown(
        f"""
        <div style="opacity: 0.2; text-align: center; margin-top: -106px;">
            <img src='data:image/jpeg;base64,{image_to_base64(image1)}' style='max-width:105%; display:block; margin:auto;'>
        </div>
        """,
        unsafe_allow_html=True
    )

# File Upload
uploaded_file = st.sidebar.file_uploader("Upload your trades (CSV/Excel)", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        # Read the uploaded file content and decode it
        raw_text = uploaded_file.getvalue().decode("utf-8")
        
        # Filter out the unwanted text
        cleaned_text = "\n".join([
            line for line in raw_text.splitlines()
            if not line.startswith("The data provided is for informational purposes only")
        ])
        
        # Convert cleaned text to DataFrame
        data = pd.read_csv(io.StringIO(cleaned_text))
    else:
        # For Excel files, directly read the data
        data = pd.read_excel(uploaded_file)


    # Debug: Ensure data is loaded
    if data is None or data.empty:
        st.error("Uploaded file is empty or could not be read. Please upload a valid file.")
    #else:
        #st.write("### Raw Data")
        #st.dataframe(data)

    # Preprocess Data
    data['Amount'] = data['Amount'].apply(parse_amount)
    data['Activity Date'] = pd.to_datetime(data['Activity Date'], errors='coerce').dt.date
    data['Description'] = data['Description'].astype(str)

    # Extract Expiry Date and Add to Data
    data['Expiry Date'] = data['Description'].apply(extract_expiry_date)

    # Separate STO and BTC Transactions
    sto_data = data[data['Trans Code'] == 'STO']
    btc_data = data[data['Trans Code'] == 'BTC']

    # Debug: Ensure STO and BTC data are valid
    #st.write("### STO Transactions")
    #st.dataframe(sto_data)
    #st.write("### BTC Transactions")
    #st.dataframe(btc_data)

    # Group STO transactions and include Quantity
    sto_grouped = sto_data.groupby(['Description', 'Instrument']).agg({
        'Amount': 'sum',
        'Quantity': 'sum',  # Include Quantity
        'Activity Date': 'min',  # Earliest STO date
        'Expiry Date': 'first'
    }).reset_index().rename(columns={'Amount': 'STO Amount', 'Activity Date': 'STO Date'})

    # Group BTC transactions and include Quantity
    btc_grouped = btc_data.groupby(['Description', 'Instrument']).agg({
        'Amount': 'sum',
        'Quantity': 'sum',  # Include Quantity
        'Activity Date': 'max'  # Latest BTC date
    }).reset_index().rename(columns={'Amount': 'BTC Amount', 'Activity Date': 'BTC Date'})

    btc_grouped = btc_data.groupby(['Description', 'Instrument']).agg({
        'Amount': 'sum',
        'Activity Date': 'max'  # Latest BTC date
    }).reset_index().rename(columns={'Amount': 'BTC Amount', 'Activity Date': 'BTC Date'})

   
    # Merge STO and BTC Transactions
    merged_data = pd.merge(
        sto_grouped,
        btc_grouped,
        on=['Description', 'Instrument'],
        how='outer',
        suffixes=('_STO', '_BTC')
    )

    # Fill Missing Values for Quantity
    merged_data['Quantity'] = merged_data['Quantity'].fillna(0).astype(int)
    merged_data['STO Amount'] = merged_data['STO Amount'].fillna(0)
    merged_data['BTC Amount'] = merged_data['BTC Amount'].fillna(0)

    # Debug: Ensure merged data is valid
    if merged_data is None or merged_data.empty:
        st.error("Merged data is empty. Ensure your input data has matching STO and BTC transactions.")
    #else:
        #st.write("### Merged Data")
        #st.dataframe(merged_data)

    # Fill Missing Values
    merged_data['STO Amount'] = merged_data['STO Amount'].fillna(0)
    merged_data['BTC Amount'] = merged_data['BTC Amount'].fillna(0)

    # Convert STO Date and BTC Date to datetime.date
    merged_data['STO Date'] = pd.to_datetime(merged_data['STO Date'], errors='coerce').dt.date
    merged_data['BTC Date'] = pd.to_datetime(merged_data['BTC Date'], errors='coerce').dt.date

    # Calculate Activity Date (minimum of STO Date and BTC Date)
    merged_data['Activity Date'] = merged_data[['STO Date', 'BTC Date']].apply(
        lambda row: min([d for d in row if pd.notna(d)]), axis=1
    )

    # Calculate Net Premium (STO - BTC, since BTC is negative, adding effectively subtracts it)
    merged_data['Net Premium'] = merged_data['STO Amount'] + merged_data['BTC Amount']

    # Add Activity Month (based on Activity Date)
    merged_data['Activity Month'] = pd.to_datetime(merged_data['Activity Date'], errors='coerce').dt.to_period('M')

    # Tag Transactions (Expired, Rolled, Closed, Open)
    merged_data = tag_transactions(merged_data)

    # Summarize Net Premium by Month
    monthly_summary = merged_data.groupby('Activity Month')['Net Premium'].sum().reset_index()

    # Add Grand Total
    grand_total = monthly_summary['Net Premium'].sum()
    monthly_summary = pd.concat(
        [monthly_summary, pd.DataFrame({'Activity Month': ['Grand Total'], 'Net Premium': [grand_total]})],
        ignore_index=True
    )

# Guard Clause: Ensure a file is uploaded before proceeding
if not uploaded_file:
    #st.warning("Please upload a valid file to proceed.")
    st.stop()

# File Processing
if uploaded_file.name.endswith('.csv'):
    raw_text = uploaded_file.getvalue().decode("utf-8")
    cleaned_text = "\n".join([
        line for line in raw_text.splitlines()
        if not line.startswith("The data provided is for informational purposes only")
    ])
    data = pd.read_csv(io.StringIO(cleaned_text))
else:
    data = pd.read_excel(uploaded_file)

# Stop if data is empty
if data.empty:
    st.error("Uploaded file is empty or invalid.")
    st.stop()

# Ensure file processing is complete before running this block
if 'monthly_summary' in locals():
    # Styled Monthly Summary Table
    styled_monthly_summary = monthly_summary.style.format({
        "Net Premium": "${:,.2f}",
        "Net After Tax": "${:,.2f}"
    }).set_properties(**{
        'font-size': '14px',
        'text-align': 'center'
    }).set_table_styles([
        {
            'selector': 'thead th',
            'props': [
                ('background-color', '#0A57C1'),
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
# Tax rate selection and "Net After Tax" calculation
col1, col2, col3 = st.columns([0.2, 0.4, 0.2])
with col2:


    st.subheader("Tax Bracket")
    tax_rate = st.select_slider(
        "Select Tax Rate to Deduct from Net Premium",
        options=[0, 0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37],
        format_func=lambda x: f"{int(x * 100)}%"
    )

# Calculate "Net After Tax"
monthly_summary['Net After Tax'] = monthly_summary['Net Premium'] * (1 - tax_rate)

# Add Grand Total for "Net After Tax"
grand_total_net_after_tax = monthly_summary[monthly_summary['Activity Month'] != 'Grand Total']['Net After Tax'].sum()
monthly_summary.loc[monthly_summary['Activity Month'] == 'Grand Total', 'Net After Tax'] = grand_total_net_after_tax

# Styled Monthly Summary Table
styled_monthly_summary = monthly_summary.style.format({
    "Net Premium": "${:,.2f}",
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

# Columns for Layout
col1, col2, col3 = st.columns([0.3, 0.4, 0.1])
# Styled Monthly Summary Table with Grand Total Highlight
def highlight_grand_total(row):
    """Apply custom style to the Grand Total row."""
    if row['Activity Month'] == 'Grand Total':
        return ['background-color: aliceblue; color: black; font-weight: bold; text-align: center'] * len(row)
    else:
        return [''] * len(row)

styled_monthly_summary = monthly_summary.style.format({
    "Net Premium": "${:,.2f}",
    "Net After Tax": "${:,.2f}"
}).set_properties(**{
    'font-size': '18px',
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
]).apply(highlight_grand_total, axis=1)

# Monthly Summary Table Display
with col1:
    st.subheader("Summary")
    st.write(styled_monthly_summary.to_html(), unsafe_allow_html=True)

    # Convert monthly_summary DataFrame to CSV for download
    summary_csv = monthly_summary.to_csv(index=False)
    st.download_button(
        label="Download Summary CSV",
        data=summary_csv,
        file_name="monthly_summary.csv",
        mime="text/csv"
    )

# Bar Chart Function
def plot_monthly_premium(monthly_summary):
    # Remove "Grand Total" for visualization
    chart_data = monthly_summary[monthly_summary['Activity Month'] != 'Grand Total']
    chart_data['Activity Month'] = chart_data['Activity Month'].astype(str)

    fig, ax = plt.subplots(figsize=(16, 12))
    width = 0.35
    x = range(len(chart_data['Activity Month']))

    # Bar for Net Premium
    ax.bar(
        [i - width / 2 for i in x],
        chart_data['Net Premium'],
        width=width,
        label="Net Premium",
        color='navy',
        edgecolor='black'
    )

    # Bar for Net After Tax
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

    # Adjust Y-axis limits dynamically to include both positive and negative values
    min_value = min(chart_data['Net Premium'].min(), chart_data['Net After Tax'].min())
    max_value = max(chart_data['Net Premium'].max(), chart_data['Net After Tax'].max())
    ax.set_ylim(min_value * 1.2 if min_value < 0 else 0, max_value * 1.2 if max_value > 0 else 0)

    ax.axhline(0, color='black', linewidth=0.8)  # Add horizontal line at y=0 for reference
    ax.set_xticks(x)
    ax.set_xticklabels(chart_data['Activity Month'], rotation=45, ha='right')

    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='x', labelsize=20)

    # Remove unnecessary spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Increase the font size of the legend
    ax.legend(fontsize=18, loc='upper left')  # Adjust fontsize and location as needed

    st.pyplot(fig)

# Monthly Premium Bar Chart Display
with col2:
    st.subheader("Monthly Net Premium")
    plot_monthly_premium(monthly_summary)

# Two New Columns for Pie Charts
col1,col2,col3 = st.columns([0.3, 0.5, 0.5])

# Pie Chart for Open Positions by Quantity
with col2:
    # Dropdown for Month Selection
    #st.subheader("Select Month for Distribution")
    selected_month = st.selectbox(
        "Select Month",
        options=monthly_summary[monthly_summary['Activity Month'] != 'Grand Total']['Activity Month']
            .astype(str).unique(),
        index=0,
        key="month_filter_pie_col1"
    )

    # Filter the data based on the selected month
    filtered_data = merged_data[
        pd.to_datetime(merged_data['Activity Month'].astype(str)) == pd.to_datetime(selected_month)
    ]
# Two New Columns for Pie Charts
pie_col1, pie_col2 = st.columns([0.5, 0.5])

# Pie Chart for Open Positions by Quantity
with pie_col1:
    # Filter open positions for the selected month
    open_positions = filtered_data[filtered_data['Status'] == 'Open']

    # Calculate the grand total open positions by Quantity across all months
    grand_total_quantity = open_positions['Quantity'].sum()

    if not open_positions.empty and grand_total_quantity > 0:
        # Group open positions for the selected month by Instrument and sum Quantity
        stock_distribution = open_positions.groupby('Instrument')['Quantity'].sum().reset_index()

        # Calculate percentage relative to the grand total open quantity
        stock_distribution['Percentage'] = (stock_distribution['Quantity'] / grand_total_quantity) * 100

        # Function to display both percentage and numeric values
        def autopct_format(pct, all_vals):
            absolute = int(round(pct / 100. * sum(all_vals)))
            return f"{pct:.1f}%\n({absolute})"

        # Custom blue-red-green color palette
        fancy_colors = [
            "#4E79A7", "#1f77b4", "#59A14F", "#E15759", "#36A2EB",
            "#0A57C1", "#1367D8", "#FF9D9A", "#1367D8", "#BAB0AC"
        ]

        # Add a glow effect as a lighter background pie
        glow_colors = [mcolors.to_rgba(c, alpha=0.3) for c in fancy_colors[:len(stock_distribution)]]

        # Plotting the glow effect as a background pie
        fig1, ax1 = plt.subplots(figsize=(3, 3))
        ax1.pie(
            stock_distribution['Percentage'],
            labels=None,
            startangle=140,
            radius=1.05,  # Slightly larger radius for glow
            colors=glow_colors,
            wedgeprops={'linewidth': 0}  # No border on glow
        )

        # Main pie chart with shadow
        wedges, texts, autotexts = ax1.pie(
            stock_distribution['Percentage'],
            labels=stock_distribution['Instrument'],
            autopct=lambda pct: autopct_format(pct, stock_distribution['Quantity']),
            startangle=140,
            colors=fancy_colors[:len(stock_distribution)],
            wedgeprops={'edgecolor': 'white', 'linewidth': 1},
            
            textprops={'color': 'black', 'fontsize': 6}  # Instrument names in black
        )

        # Label formatting
        plt.setp(autotexts, color='white', size=6)
        plt.title(f"Open Positions ({selected_month})", fontsize=8)
        st.pyplot(fig1)
    else:
        st.write("No open positions for the selected month.")
# Pie Chart for Positive Premium Distribution by Stock
with pie_col2:
    #st.subheader(f"(Closed/Expired) ({selected_month})")

    # Filter closed and expired transactions for the selected month with positive premium
    positive_premium_positions = filtered_data[
        (filtered_data['Status'].isin(['Closed', 'Expired'])) & (filtered_data['Net Premium'] > 0)
    ]
    stock_positive_premium = (
        positive_premium_positions.groupby('Instrument')['Net Premium']
        .sum()
        .reset_index()
    )

    # Plot Pie Chart
    if not stock_positive_premium.empty:
        fig2, ax2 = plt.subplots(figsize=(4, 4))
        wedges, texts, autotexts = ax2.pie(
            stock_positive_premium['Net Premium'],
            labels=stock_positive_premium['Instrument'],
            autopct='%1.1f%%',
            startangle=140,
            colors=plt.cm.Paired.colors,
            textprops={'color': 'black', 'fontsize': 5}  # Instrument names in black
        )

        plt.setp(autotexts, color='white', size=6)
        plt.title(f"Closed/Expired Premium ({selected_month})", fontsize=9)
        plt.setp(autotexts, size=8)
        st.pyplot(fig2)
    else:
        st.write("No closed/expired transactions with positive premium for the selected month.")

# Reorder columns to make 'Activity Month' the first column
columns = ['Activity Month'] + [col for col in merged_data.columns if col != 'Activity Month']
merged_data = merged_data[columns]

# Sort the detailed transactions in descending order by Activity Date
sorted_transactions = merged_data.sort_values(by='Activity Date', ascending=False)

# Style the sorted transactions table
styled_transactions = sorted_transactions.style.format({
    "STO Amount": "${:,.2f}",
    "BTC Amount": "${:,.2f}",
    "Net Premium": "${:,.2f}",
    "Activity Date": lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else '',
    "BTC Date": lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else ''  # Display blank instead of NaT
}).set_properties(**{
    'font-size': '15px',
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

# Display the sorted and styled transactions table
st.write("### Detailed Transactions")
st.write(styled_transactions.to_html(), unsafe_allow_html=True)

# Add a Download Button for the Detailed Transactions Table
detailed_csv = sorted_transactions.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Detailed Transactions CSV",
    data=detailed_csv,
    file_name="detailed_transactions.csv",
    mime="text/csv"
)
