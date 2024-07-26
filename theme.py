def run_theme_app(custom_css):
    import streamlit as st
    import pandas as pd
    import yfinance as yf
    from datetime import datetime, timedelta
    import plotly.graph_objects as go

    # Define a function to create hint text
    def hint(text):
        return f"<span title='{text}'><span style='font-size: 18px; top: -15px; bottom: -25px; color: red; border-radius: 50%; border: 2px solid grey; padding: 0.5px 8px;'>?</span></span>"

    # Define custom CSS for the HTML table outside the function
    st.markdown("""
        <style>
        .title-container {
            border-top: 1.0px solid #082C9C;  
            border-bottom: 1.0px solid #082C9C; 
            padding: 0.1px;
            background-color: #CEDDF1;
            text-align: center;
            margin-top: 12px; margin-bottom: 4px
         }
         /* Hide the form border */
         .stForm > div > div:first-child {
             border: none;
         }
         </style>
         """, unsafe_allow_html=True)

    # Create a title for the chart with the specified style
    hint_text = hint("This section allows you to view the performance of a mix of different stocks based on the industry and sector. These are based on research that looks at some of the best performing mutual funds which use similar holdings in their portfolio. Return is a theme’s performance expressed as a percentage change in its price for the past 365 day and the cumulative return is based on the weights assigned ")

    st.markdown(f'<div class="title-container" style="margin-top: -40px; "><h3 style="color: navy; font-size: 30px; margin-top: 4px;">Sector Analysis {hint_text}</h3></div>', unsafe_allow_html=True)

    # Function to fetch stocks and weights based on selected theme
    def fetch_stocks_weights(theme):
        theme_to_filename = {
            'AI': 'mediacsv.csv',
            'Berkshire': 'Berkshire.csv',
            'Blockchain': 'block.csv',
            'Cloud Service': 'cloud.csv',
            'Custom': 'customstock.csv',
            'Fintech': 'fintech.csv',
            'Gaming': 'games.csv',
            'Health and Fitness': 'fitness.csv',
            'Home and Improvement': 'homeimp.csv',
            'Magnificent 7': 'mag7.csv',
            'Online Services': 'online.csv',
            'Scion Capital': 'Scion Capital.csv',
            'Create Custom': None  # Add an option for creating a custom theme
        }
        filename = theme_to_filename.get(theme)
        if filename:
            df = pd.read_csv(filename)
            return df['Stock'].tolist(), df['Weight'].tolist(), df['Name'].tolist()
        elif theme == 'Create Custom':
            return [], [], []  # Return empty lists for custom theme
        else:
            st.error("Invalid theme selected")
            return [], [], []

    # Function to calculate return and fetch prices
    def calculate_return_and_fetch_prices(stocks, weights, start_date):
        stock_data = pd.DataFrame()
        latest_prices = {}
        valid_stocks = []
        valid_weights = []

        for stock, weight in zip(stocks, weights):
            try:
                stock_df = yf.download(stock, start=start_date, end=datetime.now())['Adj Close']
                if not stock_df.empty:  # Check if stock_df is not empty
                    stock_data = stock_data.join(stock_df.pct_change().rename(stock), how='outer')
                    latest_prices[stock] = stock_df.iloc[-1]
                    valid_stocks.append(stock)
                    valid_weights.append(weight)
                else:
                    st.warning(f"No data available for {stock}. Skipping.")
            except Exception as e:
                st.error(f"Error downloading data for {stock}: {e}. Skipping.")

        if stock_data.empty:  # Check if stock_data is empty
            return pd.Series(), {}, []

        weighted_returns = stock_data[valid_stocks] * valid_weights
        overall_daily_return = weighted_returns.sum(axis=1)
        cumulative_return = overall_daily_return.cumsum()

        return cumulative_return, latest_prices, valid_weights


    selected_theme = 'AI and ML'  # Default theme

    # Sidebar with a dropdown menu for selecting the theme and period
    selected_theme = st.sidebar.selectbox("Select Sector", ['AI', 'Berkshire', 'Blockchain', 'Cloud Service', 'Custom', 'Fintech', 'Gaming', 'Health and Fitness', 'Home and Improvement', 'Magnificent 7', 'Online Services', 'Scion Capital', 'Create Custom'])
    selected_period = st.sidebar.selectbox("Select Period", ['1YR', '6MO', '3MO', '1MO', '1WK'])

    # Define the start date based on the selected period
    period_dict = {'1YR': 365, '6MO': 180, '3MO': 90, '1MO': 30, '1WK': 7}
    start_date = datetime.now() - timedelta(days=period_dict[selected_period])

    # Fetch stocks and weights based on the selected theme
    stocks, weights, company_names = fetch_stocks_weights(selected_theme)

    # If the user selects "Create Custom", provide an interface to select stocks
    if selected_theme == 'Create Custom':
        with st.expander("Select Stocks for Custom Theme"):
            tickers_df = pd.read_csv('tickers.csv')  # Assuming tickers.csv has columns 'Symbol' and 'Name'
            all_stocks = tickers_df.set_index('Name')['Symbol'].to_dict()  # Create a dictionary mapping full names to symbols
            selected_names = st.multiselect("Select stocks:", list(all_stocks.keys()), default=[])  # Empty default selection
            selected_stocks = [all_stocks[name] for name in selected_names]  # Get the symbols for the selected full names
            
            stocks = selected_stocks
            if len(selected_stocks) > 0: 
                weights = [1.0 / len(selected_stocks)] * len(selected_stocks)  # Equal weights for simplicity
            else:
                weights=[]
            company_names = selected_names  # Use full names for display

    # Calculate return and fetch prices using default weights
    cumulative_return, latest_prices, valid_weights = calculate_return_and_fetch_prices(stocks, weights, start_date)

    col1, col2, col3, col4 = st.columns([0.4, 0.4, 0.4, 0.3])

    with col1:
        hint_text = hint("This chart shows the performance of the sector with the default assigned weights based on an average holding that most funds have assigned to the companies which shows the performance of the sector is better than S&P 500")

    # Corrected st.markdown line
        st.markdown(f"<h3 style='font-size:25px; color: navy; text-align: right;'>Weights & Return Rate % {hint_text}</h3>", unsafe_allow_html=True)

        # Create and display the chart at the top
        sp500_data = yf.download('^GSPC', start=start_date, end=datetime.now())['Adj Close']
        fig = go.Figure()



        if not cumulative_return.empty:
            fig.add_trace(go.Scatter(x=cumulative_return.index, y=cumulative_return * 100, mode='lines', name='Cumulative Return (Default)'))
            most_recent_cumulative_return = cumulative_return.iloc[-1] * 100
        else:
            most_recent_cumulative_return = 0  # Set to 0 if cumulative_return is empty

        fig.add_trace(go.Scatter(x=sp500_data.index, y=sp500_data, mode='lines', name='S&P 500', yaxis='y2', line=dict(color='red', dash='dot')))

        fig.update_layout(
            title={
                'text': f"{selected_period} Return for {selected_theme}: ({most_recent_cumulative_return:.2f}%)",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            legend=dict(x=0.2, y=1.1, orientation='h'),
            yaxis2=dict(title='S&P 500', overlaying='y', side='right'),
            xaxis_rangeslider_visible=False,
            xaxis=dict(tickformat="%b %y", hoverformat="%Y-%m-%d"),
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            yaxis2_showgrid=False,
        )
        st.plotly_chart(fig)

        # Display the price change table using default weights
        price_a_year_ago = {stock: yf.download(stock, start=start_date, end=datetime.now())['Adj Close'].iloc[0] for stock in stocks}
        price_change_percentage = [(latest_prices[stock] - price_a_year_ago[stock]) / price_a_year_ago[stock] * 100 for stock in stocks]
        df_price_change = pd.DataFrame({
            "Company": company_names,
            "Latest Price": [round(latest_prices[stock], 2) for stock in stocks],
            "Weight(%)": [round(weight * 100, 2) for weight in weights],
            f"Price {selected_period} Ago": [round(price_a_year_ago[stock], 2) for stock in stocks],
            "Price Change": [round(latest_prices[stock] - price_a_year_ago[stock], 2) for stock in stocks],
            "Price Change(%)": [round(change, 2) for change in price_change_percentage]
        })
        html_df = df_price_change.to_html(index=False, border=0)
        html_table = custom_css + html_df
        st.markdown(html_table, unsafe_allow_html=True)

   

 
    with col3:
        hint_text = hint("Enter a dollar amount you want to invest then click Adjust weights to display the drop down for each stock in the sector. You can use the slider to adjust the weights and then click on update weights. The chart will then show you how your  distribution would have performed over a 1 year period compared to default distribution on the left as well as against the S&P 500. There will be a warning message if your distribution is over 100% of the amount you entered.")

    # Corrected st.markdown line
        st.markdown(f"<h3 style='font-size:25px; color:navy; text-align: center;'>Custom Weights & Investment($) {hint_text} </h3>", unsafe_allow_html=True)



        st.markdown("""
          <style>
          .big-font {
              font-size: 20px !important;
          }
          </style>
          """, unsafe_allow_html=True)

        # Use the custom class for the number input label
        st.markdown('<p class="big-font">Enter Investment Amount</p>', unsafe_allow_html=True)
        dollar_amount = st.number_input("", min_value=0.0, value=1000.0, step=100.0)
        # Initialize selected_stocks and selected_weights with default values
        selected_stocks = [(stock, name) for stock, name in zip(stocks, company_names)]
        selected_weights = weights.copy()

        # Wrap the input fields and the button in a form
        with st.form(key='custom_weights_form'):
            with st.expander("Adjust Weights", expanded=False):
                new_selected_stocks = []
                total_weight = 0  # Initialize total weight
                for i, (stock, weight, name) in enumerate(zip(stocks, weights, company_names)):
                    checkbox_key = f"checkbox_{stock}"
                    slider_key = f"slider_{stock}"

                    # Create a checkbox and slider for each stock
                    checkbox_value = st.checkbox(f"Select {name}", key=checkbox_key, value=True)
                    if checkbox_value:
                        new_weight = st.slider(f"{name} Weight", min_value=0.0, max_value=1.0, value=weight, step=0.01, key=slider_key)
                        selected_weights[i] = new_weight
                        new_selected_stocks.append((stock, name))
                        total_weight += new_weight
                    else:
                        selected_weights[i] = 0.0

                # Display a message if the total weight reaches 100%
            if round(total_weight, 2) == 1.0:  # Use round() to handle floating-point arithmetic issues
                st.success("Total weight is 100%.")
            elif total_weight > 1.0:
                st.warning("Total weight exceeds 100%.")
            else:
                st.info(f"Total weight is {total_weight * 100:.2f}%.Adjust the weights to reach 100%.")

            distribution_choice = st.radio("Select Stocks", ['All', 'Top 10'], key="radio_distribution_choice")

            # Submit button for the form
            submit_button = st.form_submit_button(label='Update Weights')

         
        # Check if the form has been submitted
        if submit_button:
            selected_stocks = new_selected_stocks
            if distribution_choice == 'Top 10':
                sorted_stocks = sorted(zip(selected_stocks, selected_weights), key=lambda x: x[1], reverse=True)[:10]
                selected_stocks, selected_weights = zip(*sorted_stocks)
            else:
                selected_stocks, selected_weights = [stock for stock, weight in zip(selected_stocks, selected_weights) if weight > 0], [weight for stock, weight in zip(selected_stocks, selected_weights) if weight > 0]

            # Calculate return and fetch prices using adjusted weights
            adjusted_cumulative_return, _, _ = calculate_return_and_fetch_prices([stock for stock, _ in selected_stocks], selected_weights, start_date)

            # Update the custom_fig with the new data
            custom_fig = go.Figure()
            custom_fig.add_trace(go.Scatter(x=cumulative_return.index, y=cumulative_return * 100, mode='lines', name='Cumulative Return (Default)'))
            custom_fig.add_trace(go.Scatter(x=adjusted_cumulative_return.index, y=adjusted_cumulative_return * 100, mode='lines', name='Cumulative Return (Custom)'))
            custom_fig.add_trace(go.Scatter(x=sp500_data.index, y=sp500_data, mode='lines', name='S&P 500', yaxis='y2', line=dict(color='red', dash='dot')))

            if not adjusted_cumulative_return.empty:
                most_recent_cumulative_return2 = adjusted_cumulative_return.iloc[-1] * 100
                custom_fig.update_layout(
                    title={
                        'text': f"{selected_period} Return for {selected_theme} (Custom Weights): ({most_recent_cumulative_return2:.2f}%)",
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 20}
                    },
                    xaxis_title="Date",
                    yaxis_title="Cumulative Return (%)",
                    legend=dict(x=0.2, y=1.1, orientation='h'),
                    yaxis2=dict(title='S&P 500', overlaying='y', side='right'),
                    xaxis_rangeslider_visible=False,
                    xaxis_showgrid=False,
                    yaxis_showgrid=False,
                    yaxis2_showgrid=False,
                )
                st.plotly_chart(custom_fig)
            else:
                st.warning("No data available for the selected stocks and weights.")

        # Recalculate the dollar distribution table whenever the dollar amount or weights change
        dollar_distribution = [dollar_amount * weight for weight in selected_weights]
        shares = [dollar_distribution[i] / latest_prices[selected_stocks[i][0]] for i in range(len(selected_stocks))]
        price_change = [round(latest_prices[stock] - price_a_year_ago[stock], 2) for stock, _ in selected_stocks]
        price_change_percentage = [(latest_prices[stock] - price_a_year_ago[stock]) / price_a_year_ago[stock] * 100 for stock, _ in selected_stocks]

        df_dollar_distribution = pd.DataFrame({
            "Company": [name for _, name in selected_stocks],
            "Latest Price": [round(latest_prices[stock], 2) for stock, _ in selected_stocks],
            "Weight(%)": [round(weight * 100, 2) for weight in selected_weights],
            "Dollar Distribution": [round(dist, 2) for dist in dollar_distribution],
            "Shares": [round(share, 2) for share in shares],
            f"Price {selected_period} Ago": [round(price_a_year_ago[stock], 2) for stock, _ in selected_stocks],
            "Price Change": price_change,
            "Price Change(%)": [round(change, 2) for change in price_change_percentage]
        })
        html_df = df_dollar_distribution.to_html(index=False, border=0)
        html_table = custom_css + html_df
        st.markdown(html_table, unsafe_allow_html=True)

if __name__ == "__main__":
    custom_css = ""  # Add your custom CSS here
    run_theme_app(custom_css)
