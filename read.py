import streamlit as st
import pandas as pd
from datetime import datetime, date
import plotly.express as px
import re
import io
import os
import base64
import numpy as np
from PIL import Image

# ----------------------------------------------------------------------------
# Streamlit App Setup
# ----------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Trade Analysis Dashboard")

def image_to_base64(image):
    """
    Convert a PIL Image to a base64 string.
    """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# ----------------------------------------------------------------------------
# Custom CSS for the sidebar and overall app
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            background-color: #C2E0FF;
        }
        [data-testid="stSidebar"] .block-container {
            font-size: 14px;
        }
        /* Adjusting the main content font size */
        .block-container {
            font-size: 16px;
        }
        /* Custom styling for download buttons */
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

# ----------------------------------------------------------------------------
# Initialize Sidebar State
# ----------------------------------------------------------------------------
if "sidebar_hidden" not in st.session_state:
    st.session_state.sidebar_hidden = False  # Sidebar is visible by default

# ----------------------------------------------------------------------------
# Custom CSS for Sidebar Visibility
# ----------------------------------------------------------------------------
def apply_sidebar_visibility():
    """
    Applies CSS dynamically to hide or show the sidebar based on session state.
    """
    if st.session_state.sidebar_hidden:
        # Hide the sidebar
        st.markdown(
            """
            <style>
                [data-testid="stSidebar"] {
                    display: none; /* Completely hides the sidebar */
                }
                [data-testid="collapsedControl"] {
                    display: none; /* Hides the toggle arrow */
                }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Show the sidebar (default behavior)
        st.markdown(
            """
            <style>
                [data-testid="stSidebar"] {
                    display: block; /* Shows the sidebar */
                }
                [data-testid="collapsedControl"] {
                    display: block; /* Shows the toggle arrow */
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

# ----------------------------------------------------------------------------
# Apply Sidebar Visibility Dynamically
# ----------------------------------------------------------------------------
apply_sidebar_visibility()

# ----------------------------------------------------------------------------
# Toggle Sidebar Functionality
# ----------------------------------------------------------------------------
# Dynamically set button label
button_label = "Show Sidebar" if st.session_state.sidebar_hidden else "Hide Sidebar"

if st.button(button_label):
    # Toggle sidebar visibility in session state
    st.session_state.sidebar_hidden = not st.session_state.sidebar_hidden
    # Re-apply the sidebar visibility
    apply_sidebar_visibility()

# ----------------------------------------------------------------------------
# Main Content
# ----------------------------------------------------------------------------

def parse_amount(amount):
    """
    Clean monetary strings like '$1,234.56' or '(1,234.56)'.
    Returns a float sum of any numeric pieces found.
    """
    if pd.isna(amount):
        return 0.0
    text = str(amount).replace('$', '').replace(',', '').replace('(', '-').replace(')', '')
    matches = re.findall(r'[+-]?\d+(?:\.\d+)?', text)
    if not matches:
        return 0.0
    return sum(float(m) for m in matches)

def extract_expiry_date(description):
    """
    Extract expiration date (e.g., '1/17/2025') from text.
    Return a datetime if found, else None.
    """
    if pd.isna(description):
        return None
    match = re.search(r'\b(\d{1,2}/\d{1,2}/\d{4})\b', str(description))
    if match:
        try:
            return datetime.strptime(match.group(1), '%m/%d/%Y')
        except ValueError:
            return None
    return None

def parse_option_type(description):
    """
    Return 'Call' if description has 'call', 'Put' if 'put', else 'Other'.
    """
    if pd.isna(description):
        return 'Other'
    desc_lower = description.lower()
    if 'call' in desc_lower:
        return 'Call'
    elif 'put' in desc_lower:
        return 'Put'
    return 'Other'

def parse_strike_price(description):
    """
    Extract a float from a pattern like '$142.00'.
    Return 0.0 if not found.
    """
    if pd.isna(description):
        return 0.0
    match = re.search(r'\$(\d+(?:\.\d+)?)', str(description))
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return 0.0
    return 0.0

# ----------------------------------------------------------------------------
# Tagging Transactions
# ----------------------------------------------------------------------------
def tag_transactions(merged_data):
    """
    Tag each row as Expired, Closed, Rolled, Open, or Unknown.
    Checks real date comparisons and presence of BTC/Expiry.
    """
    def assign_status(row):
        current_date = date.today()

        # Convert Expiry Date to a date if it's a datetime
        expiry_date = None
        if pd.notna(row['Expiry Date']):
            if isinstance(row['Expiry Date'], datetime):
                expiry_date = row['Expiry Date'].date()
            elif hasattr(row['Expiry Date'], 'date'):
                expiry_date = row['Expiry Date'].date()
            else:
                expiry_date = row['Expiry Date']

        # Condition for "Open"
        if (
            pd.notna(row['STO Date']) and
            pd.isna(row['BTC Date']) and
            (expiry_date is not None) and
            (expiry_date > current_date)
        ):
            return "Open"

        # Condition for "Expired"
        if (
            pd.isna(row['BTC Date']) and
            (expiry_date is not None) and
            (expiry_date < current_date)
        ):
            return "Expired"

        # Condition for "Closed": BTC date present, BTC Price < 2
        if (
            pd.notna(row['BTC Date']) and
            abs(row['BTC($)'] / 100) < 2
        ):
            return "Closed"

        # Condition for "Rolled": BTC & STO exist, BTC($) > 2
        if (
            pd.notna(row['BTC Date']) and
            pd.notna(row['STO Date']) and
            abs(row['BTC($)']) > 2
        ):
            return "Rolled"

        return "Open"

    merged_data['Status'] = merged_data.apply(assign_status, axis=1)
    return merged_data

def compute_quantity(row):
    """
    If STO($)!=0 => sto_amount/(sto_price*100).
    Else if BTC($)!=0 => btc_amount/(btc_price*100).
    """
    if row['STO($)'] != 0:
        amt = row['STO($)']
        price = row['STO Price']
    elif row['BTC($)'] != 0:
        amt = row['BTC($)']
        price = row['BTC Price']
    else:
        return 0

    if pd.isna(price) or price == 0:
        return 0
    return amt / (price * 100)

def calculate_premium(row):
    """
    Because BTC($) is negative from the raw data,
    we effectively do STO($) - abs(BTC($)).
    """
    return row['STO($)'] - abs(row['BTC($)'])

def generate_unique_activity_months(merged_data):
    """
    Generate a sorted list of unique activity months in 'YYYY-MM' format.
    Sorted in descending order.
    """
    unique_months = merged_data['Activity Month'].dropna().unique().tolist()
    unique_months = sorted(unique_months, reverse=True)
    return unique_months

def generate_unique_expiry_months(merged_data):
    """
    Generate a sorted list of unique expiry months in 'YYYY-MM' format.
    Sorted in descending order.
    """
    unique_months = merged_data['Expiry Month'].dropna().unique().tolist()
    unique_months = sorted(unique_months, reverse=True)
    return unique_months

# ----------------------------------------------------------------------------
# Display header images and title
# ----------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns([0.6, 0.5, 0.5, 0.5])
with col2:
    try:
        image1 = Image.open('coined.jpeg')
        st.markdown(
            f"""
            <div style="text-align: center; margin-top: -130px;">
                <img src='data:image/jpeg;base64,{image_to_base64(image1)}' style='max-width:120%; display:block; margin:auto;'>
            </div>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.warning("Header image 'coined.jpeg' not found. Please ensure the image is in the correct directory.")

# ----------------------------------------------------------------------------
# File Upload + Data Preprocessing
# ----------------------------------------------------------------------------
uploaded_file = st.sidebar.file_uploader("Upload your trades (CSV/Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Hide the sidebar by clearing its content
    st.sidebar.empty()

    # Get today's date in the desired format (e.g., 'dec29')
    current_date_str = datetime.now().strftime("%b%d").lower()

    # Generate the new filename
    new_filename = f"{current_date_str}.csv"

    # Save the uploaded file with the new name
    sanitized_path = os.path.join('/tmp', new_filename)  # Use '/tmp' or your desired directory
    with open(sanitized_path, 'wb') as f:
        f.write(uploaded_file.read())

    st.write(f"**File Imported:** {new_filename}")

    # Load the data based on file type
    if new_filename.endswith('.csv'):
        data = pd.read_csv(sanitized_path, on_bad_lines="skip")
    else:
        data = pd.read_excel(sanitized_path)

    # 2) Strip whitespace from columns
    data.columns = data.columns.str.strip()

    # 3) Convert blank BTC Date => NaN => datetime
    if 'BTC Date' in data.columns:
        data['BTC Date'] = data['BTC Date'].replace('', pd.NA)
        data['BTC Date'] = pd.to_datetime(data['BTC Date'], errors='coerce')

    # Parse numeric fields
    if 'Amount' in data.columns:
        data['Amount'] = data['Amount'].apply(parse_amount)
    if 'Price' in data.columns:
        data['Price'] = data['Price'].apply(parse_amount)
    if 'Quantity' in data.columns:
        data['Quantity'] = data['Quantity'].apply(parse_amount)

    # -------------------------------------------------------------------
    # Parse Option Type and Strike Price directly
    # -------------------------------------------------------------------
    if 'Description' in data.columns:
        data['Option Type'] = data['Description'].apply(parse_option_type)
        data['Strike Price'] = data['Description'].apply(parse_strike_price)
        data['Expiry Date'] = data['Description'].apply(extract_expiry_date)
    else:
        st.error("The uploaded data does not contain a 'Description' column.")
        st.stop()

    # -------------------------------------------------------------------
    # Consolidate partial fills
    # -------------------------------------------------------------------
    unique_cols = ["Instrument", "Trans Code", "Activity Date", "Description"]
    agg_dict = {
        "Amount": "sum",
        "Price": "mean",
        "Option Type": "first",
        "Strike Price": "first",
        "Expiry Date": "first"
    }

    data_consolidated = (
        data
        .groupby(unique_cols, as_index=False, dropna=False)
        .agg(agg_dict)
    )

    essential_cols = ['Activity Date', 'Instrument', 'Trans Code']
    data_consolidated = data_consolidated.dropna(subset=essential_cols)

    # -------------------------------------------------------------------
    # Separate STO vs BTC
    # -------------------------------------------------------------------
    sto_data = data_consolidated[data_consolidated['Trans Code'] == 'STO']
    btc_data = data_consolidated[data_consolidated['Trans Code'] == 'BTC']

    sto_agg = {
        "Amount": "sum",
        "Price": "mean",
        "Activity Date": "min",
        "Option Type": "first",
        "Strike Price": "first",
        "Expiry Date": "first"
    }
    sto_grouped = (
        sto_data
        .groupby(["Description", "Instrument"], dropna=False)
        .agg(sto_agg)
        .reset_index()
        .rename(columns={
            "Amount": "STO($)",
            "Price": "STO Price",
            "Activity Date": "STO Date"
        })
    )

    btc_agg = {
        "Amount": "sum",
        "Price": "mean",
        "Activity Date": "max"
    }
    btc_grouped = (
        btc_data
        .groupby(["Description", "Instrument"], dropna=False)
        .agg(btc_agg)
        .reset_index()
        .rename(columns={
            "Amount": "BTC($)",
            "Price": "BTC Price",
            "Activity Date": "BTC Date"
        })
    )

    # Merge STO & BTC
    merged_data = pd.merge(
        sto_grouped,
        btc_grouped,
        on=["Description", "Instrument"],
        how="outer"
    )

    # Fill missing numeric values
    merged_data['STO($)'] = merged_data['STO($)'].fillna(0)
    merged_data['BTC($)'] = merged_data['BTC($)'].fillna(0)
    merged_data['STO Price'] = merged_data['STO Price'].fillna(0)
    merged_data['BTC Price'] = merged_data['BTC Price'].fillna(0)

    # Combine the Activity Date as the earliest date between STO Date and BTC Date
    merged_data['Activity Date'] = merged_data[['STO Date', 'BTC Date']].apply(
        lambda row: min([d for d in row if pd.notna(d)]) if any(pd.notna(d) for d in row) else pd.NaT,
        axis=1
    )

    # Ensure 'Activity Date' is in datetime format
    merged_data['Activity Date'] = pd.to_datetime(merged_data['Activity Date'], errors='coerce')

    # Premium($), Tag, etc.
    merged_data['Amount'] = merged_data['STO($)'] + merged_data['BTC($)']
    merged_data['Activity Month'] = merged_data['Activity Date'].dt.strftime('%Y-%m')  # Now works correctly

    # Create 'Expiry Month' as 'YYYY-MM' string
    merged_data['Expiry Month'] = merged_data['Expiry Date'].dt.strftime('%Y-%m')

    merged_data['Quantity'] = merged_data.apply(compute_quantity, axis=1).round(0).astype(int)
    merged_data['Premium($)'] = merged_data.apply(calculate_premium, axis=1)

    # Tag transactions
    merged_data = tag_transactions(merged_data)

    # -------------- Omit 'Description' from the final table --------------
    if 'Description' in merged_data.columns:
        merged_data.drop(columns=['Description'], inplace=True)

    # -----------------------------------------------------------------------
    # Monthly Summary (sum Premium($))
    # -----------------------------------------------------------------------
    monthly_summary = (
        merged_data
        .groupby('Activity Month')['Premium($)']
        .sum()
        .reset_index()
        .rename(columns={'Premium($)': 'Net Premium'})
        .sort_values(by='Activity Month')
    )

    # ----------------------------------------------------------------------------
    # Generate Unique Months for Filtering
    # ----------------------------------------------------------------------------
    unique_expiry_months = generate_unique_expiry_months(merged_data)
    unique_activity_months = generate_unique_activity_months(merged_data)

    # ----------------------------------------------------------------------------
    # Debugging: Verify 'Activity Month' and 'Expiry Month' Formats
    # ----------------------------------------------------------------------------
    #st.write("### Unique Activity Months in Data")
    #st.write(unique_activity_months)

    #st.write("### Unique Expiry Months in Data")
    #st.write(unique_expiry_months)

    # ----------------------------------------------------------------------------
    # Layout: Tax Bracket Slider
    # ----------------------------------------------------------------------------
    tax_colA, tax_colB, tax_colC = st.columns([1, 2, 1])
    with tax_colB:
        st.subheader("Tax Bracket")
        tax_rate = st.select_slider(
            "Select Tax Rate to Deduct from Net Premium",
            options=[0, 0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37],
            format_func=lambda x: f"{int(x*100)}%"
        )

    # Apply Tax Rate
    monthly_summary['Net After Tax'] = monthly_summary['Net Premium'] * (1 - tax_rate)

    # Add Grand Total for Net Premium and Net After Tax
    grand_total_net_premium = monthly_summary['Net Premium'].sum()
    grand_total_net_after_tax = monthly_summary['Net After Tax'].sum()

    # Add Grand Total row
    grand_total_row = pd.DataFrame({
        'Activity Month': ['Grand Total'],
        'Net Premium': [grand_total_net_premium],
        'Net After Tax': [grand_total_net_after_tax]
    })

    monthly_summary = pd.concat(
        [monthly_summary, grand_total_row],
        ignore_index=True
    )

    # ----------------------------------------------------------------------------
    # Layout: Summary Section and Bar Chart
    # ----------------------------------------------------------------------------
    summary_col1, summary_col2, summary_col3 = st.columns([0.3, 0.05, 0.5])  # Adjusted column ratios
    with summary_col1:
        st.subheader("Summary")
        # Style monthly summary
        styled_monthly_summary = (
            monthly_summary
            .style
            .format({"Net Premium":"${:,.2f}","Net After Tax":"${:,.2f}"})
            .set_properties(**{'font-size':'18px','text-align':'center'})
            .set_table_styles([
                {
                    'selector':'thead th',
                    'props':[
                        ('background-color','aliceblue'),
                        ('color','black'),
                        ('font-weight','bold'),
                        ('text-align','center'),
                        ('border','1px solid #CCCCCC'),
                        ('padding','10px')
                    ]
                },
                {
                    'selector':'tbody td',
                    'props':[
                        ('padding','10px'),
                        ('border','1px solid #CCCCCC')
                    ]
                },
                {
                    'selector':'table',
                    'props':[
                        ('width','100%'),
                        ('margin','0 auto'),
                        ('border-collapse','collapse')
                    ]
                }
            ])
            .apply(lambda row: ['background-color: aliceblue; color: black; font-weight: bold; text-align:center']*len(row) if row['Activity Month'] == 'Grand Total' else ['']*len(row), axis=1)
        )
        st.write(styled_monthly_summary.to_html(), unsafe_allow_html=True)
        summary_csv = monthly_summary.to_csv(index=False)
        st.download_button(
            label="Download Summary CSV",
            data=summary_csv,
            file_name="monthly_summary.csv",
            mime="text/csv"
        )

    # Bar Chart (Using Plotly Express)
    def plot_monthly_premium(monthly_summary):
        chart_data = monthly_summary[monthly_summary['Activity Month'] != 'Grand Total']
        chart_data['Activity Month'] = chart_data['Activity Month'].astype(str)

        # Ensure 'Net After Tax' exists
        if 'Net After Tax' not in chart_data.columns:
            st.error("'Net After Tax' column is missing in chart_data.")
            return

        # Create a grouped bar chart
        try:
            fig = px.bar(
                chart_data,
                x='Activity Month',
                y=['Net Premium', 'Net After Tax'],
                labels={'value': 'Net Premium ($)', 'Activity Month': 'Month'},
                barmode='group',
                text_auto='.2s',
                height=600,
                width=800
            )
        except ValueError as ve:
            st.error(f"Plotly Error: {ve}")
            st.stop()

        # Update layout for Arial font and increase title font size
        fig.update_layout(
            font=dict(family="Arial", size=12),
            legend=dict(font=dict(family="Arial", size=12)),
            xaxis_tickangle=-45
        )

        # Update y-axis to have some padding
        min_val = min(chart_data['Net Premium'].min(), chart_data['Net After Tax'].min())
        max_val = max(chart_data['Net Premium'].max(), chart_data['Net After Tax'].max())
        fig.update_yaxes(range=[min_val * 1.2 if min_val < 0 else 0, max_val * 1.2 if max_val > 0 else 0])

        st.plotly_chart(fig, use_container_width=True)

    with summary_col3:
        st.subheader("Monthly Net Premium")
        plot_monthly_premium(monthly_summary)

    # ----------------------------------------------------------------------------
    # *** Moved Selectors to a Single Row Above Pie Charts ***
    # ----------------------------------------------------------------------------
    # Create a single row with three selectors: Select Activity Month, Select Expiry Month, Select Instrument
    selector_colA, selector_colB, selector_colC = st.columns([0.07, 0.07, 0.07])

    with selector_colA:
        # Select Activity Month
        # Extract unique months from 'Activity Month'
        unique_activity_months = unique_activity_months  # Already generated earlier

        selected_activity_month = st.selectbox(
            label="Select Activity Month",
            options=["All"] + unique_activity_months,
            index=0,
            key="activity_month_filter"
        )

    with selector_colB:
        # Select Expiry Month
        # Extract unique months from 'Expiry Month'
        unique_expiry_months = unique_expiry_months  # Already generated earlier

        selected_expiry_month = st.selectbox(
            label="Select Expiry Month",
            options=["All"] + unique_expiry_months,
            index=0,
            key="expiry_month_filter"
        )

    with selector_colC:
        # Select Instrument
        instruments = sorted(merged_data['Instrument'].unique().tolist())
        instrument_selection = st.selectbox(
            "Select Instrument",
            options=["All"] + instruments,
            index=0,
            key="instrument_filter_combined"
        )

    # ----------------------------------------------------------------------------
    # Apply Combined Filters to the Data
    # ----------------------------------------------------------------------------
    # Start with the merged data
    filtered_data_combined = merged_data.copy()

    # Apply Select Activity Month filter using datetime comparison
    if selected_activity_month != "All":
        try:
            # Split the selected month into year and month
            year, month = map(int, selected_activity_month.split('-'))
            
            # Apply the filter based on year and month
            filtered_data_combined = filtered_data_combined[
                (filtered_data_combined['Activity Date'].dt.year == year) &
                (filtered_data_combined['Activity Date'].dt.month == month)
            ]
        except Exception as e:
            st.error(f"Error filtering by Activity Month: {e}")

    # Apply Select Expiry Month filter using datetime comparison
    if selected_expiry_month != "All":
        try:
            # Split the selected expiry month into year and month
            exp_year, exp_month = map(int, selected_expiry_month.split('-'))
            
            # Apply the filter based on expiry year and month
            filtered_data_combined = filtered_data_combined[
                (filtered_data_combined['Expiry Date'].dt.year == exp_year) &
                (filtered_data_combined['Expiry Date'].dt.month == exp_month)
            ]
        except Exception as e:
            st.error(f"Error filtering by Expiry Month: {e}")

    # Apply Select Instrument filter
    if instrument_selection != "All":
        filtered_data_combined = filtered_data_combined[filtered_data_combined['Instrument'] == instrument_selection]

    # ----------------------------------------------------------------------------
    # Debugging: Verify the Number of Rows After Filtering
    # ----------------------------------------------------------------------------
    #st.write("### Number of Transactions After Filtering:", filtered_data_combined.shape[0])

    # Optionally, display a preview of the filtered data
    #st.write("### Preview of Filtered Transactions")
    #st.write(filtered_data_combined.head())

    # ----------------------------------------------------------------------------
    # Layout: Pie Charts Positioned Below the Selectors
    # ----------------------------------------------------------------------------
    pie_col1, pie_col2, pie_col3 = st.columns(3)

    # Define a custom color palette with less orange and yellow, more blue tones
    custom_colors = [
        '#008CEE',  # Royal Blue
        '#000080',  # Navy
        '#A80000',  # Cyan
        '#107C10',  # Dodger Blue
        '#094782',  # Steel Blue
        '#6495ED',  # Cornflower Blue
        '#5F9EA0',  # Cadet Blue
        '#2E8B57',  # Sea Green
    ]

    # 1) Donut Chart: Open Positions by Quantity
    with pie_col1:
        open_positions = filtered_data_combined[filtered_data_combined['Status'] == 'Open']
        total_open_qty = open_positions['Quantity'].sum()
        if not open_positions.empty and total_open_qty > 0:
            stock_distribution = (
                open_positions
                .groupby('Instrument')['Quantity']
                .sum()
                .reset_index()
            )
            stock_distribution['Percentage'] = (stock_distribution['Quantity'] / total_open_qty) * 100

            # Use the custom color palette
            color_sequence = custom_colors[:len(stock_distribution)]

            # Update the title based on selected months
            title_parts = []
            if selected_activity_month != "All":
                title_parts.append(f"Activity Month: {selected_activity_month}")
            if selected_expiry_month != "All":
                title_parts.append(f"Expiry Month: {selected_expiry_month}")
            title_month = " & ".join(title_parts) if title_parts else "All Activity and Expiry Months"

            fig1 = px.pie(
                stock_distribution,
                names='Instrument',
                values='Quantity',
                title=f"Open Positions<br>({title_month})",
                hole=0.3,  # This makes it a donut chart
                color='Instrument',
                color_discrete_sequence=color_sequence
            )

            fig1.update_traces(
                textinfo='percent+label',
                textfont=dict(size=14, family="Arial")
            )

            fig1.update_layout(
                title=dict(font=dict(family="Arial", size=18)),  # Increased font size
                legend_title_text='Instrument',
                legend=dict(font=dict(family="Arial", size=12), orientation='h', x=0.5, xanchor='center', y=-0.1),
                margin=dict(l=10, r=10, t=60, b=40)  # Adjusted top margin for larger title
            )

            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.write("No open positions for the selected filters.")

    # 2) Donut Chart: Closed/Expired Premium
    with pie_col2:
        # Filter transactions with Status 'Closed' or 'Expired'
        closed_expired = filtered_data_combined[filtered_data_combined['Status'].isin(['Closed', 'Expired'])]

        # Aggregate Premium by Instrument
        stock_positive_premium = (
            closed_expired
            .groupby('Instrument')['Premium($)']
            .sum()
            .reset_index()
        )

        if not stock_positive_premium.empty:
            # Use the custom color palette
            color_sequence = custom_colors[:len(stock_positive_premium)]

            # Update the title based on selected months
            title_parts = []
            if selected_activity_month != "All":
                title_parts.append(f"Activity Month: {selected_activity_month}")
            if selected_expiry_month != "All":
                title_parts.append(f"Expiry Month: {selected_expiry_month}")
            title_month = " & ".join(title_parts) if title_parts else "All Activity and Expiry Months"

            fig2 = px.pie(
                stock_positive_premium,
                names='Instrument',
                values='Premium($)',
                title=f"Closed/Expired Premium<br>({title_month})",
                hole=0.3,  # This makes it a donut chart
                color='Instrument',
                color_discrete_sequence=color_sequence
            )

            fig2.update_traces(
                textinfo='percent+label',
                textfont=dict(size=14, family="Arial")
            )

            fig2.update_layout(
                title=dict(font=dict(family="Arial", size=18)),  # Increased font size
                legend_title_text='Instrument',
                legend=dict(font=dict(family="Arial", size=12), orientation='h', x=0.5, xanchor='center', y=-0.1),
                margin=dict(l=10, r=10, t=60, b=40)  # Adjusted top margin for larger title
            )

            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.write("No closed/expired transactions with positive premium for the selected filters.")

    # 3) Additional Pie Chart: Calls vs Puts Distribution
    with pie_col3:
        # Calculate the distribution of Calls vs Puts within the filtered data
        calls_puts_distribution = filtered_data_combined['Option Type'].value_counts().reset_index()
        calls_puts_distribution.columns = ['Option Type', 'Count']

        # Optionally, exclude 'Other' if not relevant
        # Uncomment the following lines if needed:
        # calls_puts_distribution = calls_puts_distribution[calls_puts_distribution['Option Type'].isin(['Call', 'Put'])]

        # Calculate percentages
        total_options = calls_puts_distribution['Count'].sum()
        if total_options > 0:
            calls_puts_distribution['Percentage'] = (calls_puts_distribution['Count'] / total_options) * 100

            # Use the custom color palette
            color_sequence_cp = custom_colors[:len(calls_puts_distribution)]

            # Update the title based on selected months
            title_parts = []
            if selected_activity_month != "All":
                title_parts.append(f"Activity Month: {selected_activity_month}")
            if selected_expiry_month != "All":
                title_parts.append(f"Expiry Month: {selected_expiry_month}")
            title_month = " & ".join(title_parts) if title_parts else "All Activity and Expiry Months"

            fig3 = px.pie(
                calls_puts_distribution,
                names='Option Type',
                values='Count',
                title=f"Calls vs Puts Distribution<br>({title_month})",
                hole=0.3,  # Makes it a donut chart
                color='Option Type',
                color_discrete_sequence=color_sequence_cp
            )

            fig3.update_traces(
                textinfo='percent+label',
                textfont=dict(size=14, family="Arial")
            )

            fig3.update_layout(
                title=dict(font=dict(family="Arial", size=18)),  # Increased font size
                legend_title_text='Option Type',
                legend=dict(font=dict(family="Arial", size=12), orientation='h', x=0.5, xanchor='center', y=-0.1),
                margin=dict(l=10, r=10, t=60, b=40)  # Adjusted top margin for larger title
            )

            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.write("No option types available for the selected filters.")

    # ----------------------------------------------------------------------------
    # Detailed Transactions - sort descending order by Activity Date
    # Show specified columns in the desired order and reflect Expiry Month selection
    # ----------------------------------------------------------------------------
    if 'Activity Month' in merged_data.columns:
        # Define the desired column order as per user request
        desired_columns = [
            'Activity Date',
            'Instrument',
            'Option Type',
            'Quantity',
            'Strike Price',
            'Expiry Date',
            'STO($)',
            'BTC($)',
            'STO Price',
            'BTC Price',
            'STO Date',
            'BTC Date',
            'Status',
            'Premium($)'
        ]

        # Filter only the desired columns
        existing_desired_columns = [col for col in desired_columns if col in filtered_data_combined.columns]
        filtered_transactions = filtered_data_combined[existing_desired_columns].copy()

        # Ensure date columns are datetime objects
        date_columns = ['Activity Date', 'STO Date', 'BTC Date', 'Expiry Date']
        for col in date_columns:
            if col in filtered_transactions.columns:
                filtered_transactions[col] = pd.to_datetime(filtered_transactions[col], errors='coerce')

        # Sort the filtered transactions by 'Activity Date' descendingly
        filtered_transactions = filtered_transactions.sort_values(by='Activity Date', ascending=False)

        # Check if filtered_transactions is empty
        if filtered_transactions.empty:
            st.write("### Detailed Transactions")
            st.write("No transactions were found for the selected filters. Please adjust your selections.")
        else:
            # Final table with specified column order and sorted descendingly by Activity Date
            styled_transactions = (
                filtered_transactions
                .style
                .format({
                    "STO($)": "${:,.2f}",
                    "BTC($)": "${:,.2f}",
                    "Premium($)": "${:,.2f}",
                    "STO Price": "${:,.2f}",
                    "BTC Price": "${:,.2f}",
                    "Strike Price": "${:,.2f}",
                    "Activity Date": lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else '',
                    "STO Date": lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else '',
                    "BTC Date": lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else '',
                    "Expiry Date": lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else ''
                })
                .set_properties(**{'font-size': '15px', 'text-align': 'center'})
                .set_table_styles([
                    {
                        'selector':'thead th',
                        'props':[
                            ('background-color','aliceblue'),
                            ('color','black'),
                            ('font-weight','bold'),
                            ('text-align','center'),
                            ('border','1px solid #CCCCCC'),
                            ('padding','10px')
                        ]
                    },
                    {
                        'selector':'tbody td',
                        'props':[
                            ('padding','10px'),
                            ('border','1px solid #CCCCCC')
                        ]
                    },
                    {
                        'selector':'table',
                        'props':[
                            ('width','100%'),
                            ('margin','0 auto'),
                            ('border-collapse','collapse')
                        ]
                    }
                ])
            )

            st.write("### Detailed Transactions")
            st.write(styled_transactions.to_html(), unsafe_allow_html=True)

            # Download
            detailed_csv = filtered_transactions.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Detailed Transactions",
                data=detailed_csv,
                file_name="detailed_transactions.csv",
                mime="text/csv"
            )

    # ----------------------------------------------------------------------------
    # Additional Layout: Display Monthly Summary and Charts at the Top
    # (Optional: Move summary and charts above the filters for better UX)
    # ----------------------------------------------------------------------------
    # (This section can be adjusted based on user preference)

else:
    # Create three columns
    col1, col2, col3 = st.columns([0.2, 0.2, 0.3])

    # Display the info message in col2
    with col2:
        st.info("Please upload a CSV or Excel file to get started.")
