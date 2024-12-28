import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta
import plotly.express as px
import re
import io
import base64
from PIL import Image

# ----------------------------------------------------------------------------
# Streamlit App Setup
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Sidebar Toggle Example", layout="wide")

def image_to_base64(image):
    """
    Convert a PIL Image to a base64 string.
    """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()



import streamlit as st

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

def generate_fridays(start_year=2023, end_year=2025):
    """
    Generate a list of dates that are Fridays from January 2023 to December 2025.
    """
    fridays = []
    for year in range(start_year, end_year + 1):
        # Start from Jan 1 of the year
        date_range = pd.date_range(start=f'1/1/{year}', end=f'12/31/{year}', freq='W-FRI')
        fridays.extend(date_range)
    # Convert to date objects
    fridays = [d.date() for d in fridays]
    return fridays

# ----------------------------------------------------------------------------
# Display header images and title
# ----------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns([0.6, 0.5, 0.5, 0.5])
with col2:
    try:
        image1 = Image.open('coined.jpeg')
        st.markdown(
            f"""
            <div style="text-align: center; margin-top: -100px;">
                <img src='data:image/jpeg;base64,{image_to_base64(image1)}' style='max-width:120%; display:block; margin:auto;'>
            </div>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.warning("Header image 'coined.jpg' not found. Please ensure the image is in the correct directory.")

# ----------------------------------------------------------------------------
# File Upload + Data Preprocessing
# ----------------------------------------------------------------------------
uploaded_file = st.sidebar.file_uploader("Upload your trades (CSV/Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Hide the sidebar by clearing its content
    st.sidebar.empty()

    # 1) Load data
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file, on_bad_lines="skip")
    else:
        data = pd.read_excel(uploaded_file)

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
        data['Expiry Date'] = pd.to_datetime(data['Expiry Date'], errors='coerce').dt.date
    else:
        st.error("The uploaded data does not contain a 'Description' column.")
        st.stop()

    # -------------------------------------------------------------------
    # Consolidate partial fills
    # -------------------------------------------------------------------
    unique_cols = ["Instrument","Trans Code","Activity Date","Description"]
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

    essential_cols = ['Activity Date','Instrument','Trans Code']
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
        .groupby(["Description","Instrument"], dropna=False)
        .agg(sto_agg)
        .reset_index()
        .rename(columns={
            "Amount":"STO($)",
            "Price":"STO Price",
            "Activity Date":"STO Date"
        })
    )

    btc_agg = {
        "Amount": "sum",
        "Price": "mean",
        "Activity Date": "max"
    }
    btc_grouped = (
        btc_data
        .groupby(["Description","Instrument"], dropna=False)
        .agg(btc_agg)
        .reset_index()
        .rename(columns={
            "Amount":"BTC($)",
            "Price":"BTC Price",
            "Activity Date":"BTC Date"
        })
    )

    # Merge STO & BTC
    merged_data = pd.merge(
        sto_grouped,
        btc_grouped,
        on=["Description","Instrument"],
        how="outer"
    )

    # Fill missing numeric
    merged_data['STO($)'] = merged_data['STO($)'].fillna(0)
    merged_data['BTC($)'] = merged_data['BTC($)'].fillna(0)
    merged_data['STO Price'] = merged_data['STO Price'].fillna(0)
    merged_data['BTC Price'] = merged_data['BTC Price'].fillna(0)

    # Combine the Activity Date
    merged_data['Activity Date'] = merged_data[['STO Date','BTC Date']].apply(
        lambda row: min([d for d in row if pd.notna(d)]) if any(pd.notna(d) for d in row) else None,
        axis=1
    )

    # Premium($), Tag, etc.
    merged_data['Amount'] = merged_data['STO($)'] + merged_data['BTC($)']
    merged_data['Activity Month'] = pd.to_datetime(
        merged_data['Activity Date'], errors='coerce'
    ).dt.to_period('M')
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
    # Layout: Tax Bracket Slider
    # ----------------------------------------------------------------------------
    tax_colA, tax_colB, tax_colC = st.columns([1, 2, 1])
    with tax_colB:
        st.subheader("Tax Bracket")
        tax_rate = st.select_slider(
            "Select Tax Rate to Deduct from Net Premium",
            options=[0,0.10,0.12,0.22,0.24,0.32,0.35,0.37],
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
    summary_col1, summary_col2, summary_col3 = st.columns([0.3, 0.1, 0.5])
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
                #title='Monthly Net Premium',
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
            #title=dict(font=dict(family="Arial", size=20)),  # Increased font size
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
    # Layout: Three Columns with Selector in the Middle
    # ----------------------------------------------------------------------------
    selector_colA, selector_colB, selector_colC = st.columns([1, 2, 1])

    with selector_colB:
        # Extract unique months excluding 'Grand Total'
        months = monthly_summary[
            monthly_summary['Activity Month'] != 'Grand Total'
        ]['Activity Month'].astype(str).unique()
        
        # Convert month strings to datetime objects for proper sorting
        months_datetime = pd.to_datetime(months, format='%Y-%m', errors='coerce')
        
        # Remove any NaT values resulting from incorrect formats
        months_datetime = months_datetime.dropna()
        
        # Sort the months in descending order (most recent first)
        sorted_months_datetime = months_datetime.sort_values(ascending=False)
        
        # Convert back to string format 'YYYY-MM'
        sorted_months = sorted_months_datetime.strftime('%Y-%m').tolist()
        
        # Get the current month in 'YYYY-MM' format
        current_month = date.today().strftime('%Y-%m')
        
        # Determine the default index: current month if exists, else first month
        if current_month in sorted_months:
            default_index = sorted_months.index(current_month)
        else:
            default_index = 0  # Default to the first month if current month not found
        
        # Create the selectbox with sorted options and default selection
        selected_month = st.selectbox(
            label="",
            options=sorted_months,
            index=default_index,
            key="month_filter_pie_col1"
        )
        
        # Properly filter data based on the selected month
        filtered_data = merged_data[
            merged_data['Activity Month'].astype(str) == selected_month
        ]

    # ----------------------------------------------------------------------------
    # Layout: Pie Charts Positioned Below the Selector
    # ----------------------------------------------------------------------------
    # Create three columns for the pie charts
    pie_col1, pie_col2, pie_col3 = st.columns(3)

    # Define a custom color palette with less orange and yellow, more blue tones
    custom_colors = [
        '#008CEE',  # Royal Blue
        '#000080',  # Navy
        '#A80000',  # Cyan
        '#107C10',  # Dodger Blue
        '#094782',  # Steel Blue
        '#F17925',  # Light Sky Blue
        '#6495ED',  # Cornflower Blue
        '#5F9EA0',  # Cadet Blue
    ]

    # 1) Donut Chart: Open Positions by Quantity
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

            # Use the custom color palette
            color_sequence = custom_colors[:len(stock_distribution)]

            fig1 = px.pie(
                stock_distribution,
                names='Instrument',
                values='Quantity',
                title=f"Open Positions<br>({selected_month})",
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
            st.write("No open positions for the selected month.")

    # 2) Donut Chart: Closed/Expired Premium
    with pie_col2:
        positive_premium_positions = filtered_data[
            (filtered_data['Status'].isin(['Closed', 'Expired'])) & 
            (filtered_data['Amount'] > 0)
        ]
        stock_positive_premium = (
            positive_premium_positions
            .groupby('Instrument')['Amount']
            .sum()
            .reset_index()
        )
        if not stock_positive_premium.empty:
            # Use the custom color palette
            color_sequence = custom_colors[:len(stock_positive_premium)]

            fig2 = px.pie(
                stock_positive_premium,
                names='Instrument',
                values='Amount',
                title=f"Closed/Expired Premium<br>({selected_month})",
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
            st.write("No closed/expired transactions with positive premium for the selected month.")

    # 3) Additional Pie Chart: Calls vs Puts Distribution
    with pie_col3:
        # Calculate the distribution of Calls vs Puts
        calls_puts_distribution = merged_data['Option Type'].value_counts().reset_index()
        calls_puts_distribution.columns = ['Option Type', 'Count']

        # Optionally, exclude 'Other' if not relevant
        # Uncomment the following lines if needed:
        # calls_puts_distribution = calls_puts_distribution[calls_puts_distribution['Option Type'].isin(['Call', 'Put'])]

        # Calculate percentages
        total_options = calls_puts_distribution['Count'].sum()
        calls_puts_distribution['Percentage'] = (calls_puts_distribution['Count'] / total_options) * 100

        # Use the custom color palette
        color_sequence_cp = custom_colors[:len(calls_puts_distribution)]

        fig3 = px.pie(
            calls_puts_distribution,
            names='Option Type',
            values='Count',
            title=f"Calls vs Puts Distribution<br>({selected_month})",
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

    # ----------------------------------------------------------------------------
    # Detailed Transactions - sort descending by Activity Date
    # Omit the 'Description' column from final table
    # ----------------------------------------------------------------------------
    if 'Activity Month' in merged_data.columns:
        # Rearrange columns as per user request
        desired_columns = [
            'Activity Date',   # First column
            'Instrument',      # After Activity Date
            'Expiry Date',     # After Instrument
            'STO($)', 'BTC($)',  # STO($) and BTC($) next to each other
            'STO Price', 'BTC Price',  # STO Price and BTC Price next to each other
            'STO Date', 'BTC Date'     # STO Date and BTC Date next to each other
        ]

        # Add any other columns that are not specified
        other_columns = [col for col in merged_data.columns if col not in desired_columns]

        # Ensure the desired columns exist in merged_data
        existing_desired_columns = [col for col in desired_columns if col in merged_data.columns]

        # Rearrange the columns
        merged_data = merged_data[existing_desired_columns + other_columns]

    merged_data['Activity Date'] = pd.to_datetime(merged_data['Activity Date'], errors='coerce')
    sorted_transactions = merged_data.sort_values(by='Activity Date', ascending=False)

    # Convert STO/BTC date
    for dc in ['STO Date','BTC Date']:
        if dc in sorted_transactions.columns:
            sorted_transactions[dc] = pd.to_datetime(sorted_transactions[dc], errors='coerce')

    # ----------------------------------------------------------------------------
    # Layout: Filters Above Detailed Transactions Table
    # ----------------------------------------------------------------------------
    # Generate list of Fridays
    fridays = generate_fridays(start_year=2023, end_year=2025)

    # Add "All" to the list of fridays
    formatted_fridays = [d.strftime('%Y-%m-%d') for d in fridays]
    formatted_fridays.insert(0, "All")

    # Determine if today is Friday
    today = date.today()
    if today.weekday() == 4:  # 4 represents Friday
        today_str = today.strftime('%Y-%m-%d')
        if today_str in formatted_fridays:
            default_expiry = today_str
        else:
            default_expiry = "All"
    else:
        default_expiry = "All"

    # Create three columns for filters
    filter_col1, filter_col2, filter_col3 = st.columns([1, 2, 2])
    with filter_col1:
        st.subheader("Filters")
    with filter_col2:
        instruments = sorted_transactions['Instrument'].unique().tolist()
        instruments.sort()
        instrument_selection = st.selectbox(
            "Select Instrument",
            options=["All"] + instruments,
            index=0,
            key="instrument_filter"
        )
    with filter_col3:
        selected_friday = st.selectbox(
            "Select Expiry Date",
            options=formatted_fridays,
            index=formatted_fridays.index(default_expiry) if default_expiry in formatted_fridays else 0,
            key="date_filter"
        )
    

    # Apply filters
    filtered_transactions = sorted_transactions.copy()

    if instrument_selection != "All":
        filtered_transactions = filtered_transactions[filtered_transactions['Instrument'] == instrument_selection]

    if selected_friday != "All":
        # Convert selected_friday to date object for comparison
        try:
            selected_friday_date = datetime.strptime(selected_friday, '%Y-%m-%d').date()
            filtered_transactions = filtered_transactions[filtered_transactions['Expiry Date'] == selected_friday_date]
        except ValueError:
            st.error("Selected Expiry Date format is incorrect.")
            st.stop()

    # Check if filtered_transactions is empty
    if filtered_transactions.empty:
        st.write("No transactions were found for this date. Please choose another date!!")
    else:
        # Final table
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
                "Amount": "${:,.2f}",  # Added formatting for 'Amount'
                "Activity Date": lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else '',
                "STO Date": lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else '',
                "BTC Date": lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else ''
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
    # Custom CSS for the download button
    # ----------------------------------------------------------------------------
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

else:
    st.info("Please upload a CSV or Excel file to get started.")