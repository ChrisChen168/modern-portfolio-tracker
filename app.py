import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import os
import google.generativeai as genai

# --- Configuration ---
st.set_page_config(
    page_title="Modern Stock Portfolio",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants ---
PORTFOLIO_FILE = 'portfolio.csv'

# Theme Definitions
THEME_DARK = {
    "bg": "#0E1117",
    "card": "#262730",
    "text_main": "#FFFFFF",
    "text_sec": "#888888",
    "input_bg": "#262730",
    "input_border": "#4E4E4E",
    "input_text": "#FFFFFF",
    "accent_red": "#FF4B4B",
    "accent_green": "#00CC96"
}

THEME_LIGHT = {
    "bg": "#F0F2F6",
    "card": "#FFFFFF",
    "text_main": "#000000",
    "text_sec": "#666666",
    "input_bg": "#FFFFFF",
    "input_border": "#D1D5DB",
    "input_text": "#000000",
    "accent_red": "#FF4B4B",
    "accent_green": "#00CC96"
}

# Popular Tickers for Autocomplete
POPULAR_TICKERS = [
    "🔍 Custom Search...",
    "AAPL - Apple Inc.",
    "MSFT - Microsoft Corp.",
    "GOOGL - Alphabet Inc.",
    "AMZN - Amazon.com Inc.",
    "NVDA - NVIDIA Corp.",
    "TSLA - Tesla Inc.",
    "META - Meta Platforms Inc.",
    "BRK.B - Berkshire Hathaway",
    "LLY - Eli Lilly and Co.",
    "V - Visa Inc.",
    "TSM - Taiwan Semiconductor",
    "AVGO - Broadcom Inc.",
    "NVO - Novo Nordisk",
    "JPM - JPMorgan Chase",
    "WMT - Walmart Inc.",
    "XOM - Exxon Mobil Corp.",
    "MA - Mastercard Inc.",
    "UNH - UnitedHealth Group",
    "PG - Procter & Gamble",
    "JNJ - Johnson & Johnson",
    "ORCL - Oracle Corp.",
    "HD - Home Depot",
    "COST - Costco Wholesale",
    "BAC - Bank of America",
    "ABBV - AbbVie Inc.",
    "KO - Coca-Cola Co.",
    "PEP - PepsiCo Inc.",
    "CRM - Salesforce Inc.",
    "AMD - Advanced Micro Devices",
    "NFLX - Netflix Inc.",
    "INTC - Intel Corp.",
    "DIS - Walt Disney Co.",
    "NKE - Nike Inc.",
    "PFE - Pfizer Inc."
]

# --- Theme Management ---
if 'theme' not in st.session_state:
    st.session_state.theme = 'Dark'
if 'lang' not in st.session_state:
    st.session_state.lang = 'English'

with st.sidebar:
    with st.expander("⚙️ Settings", expanded=False):
        st.write("### Appearance")
        theme_selection = st.radio("Theme", ["Dark", "Light"], index=0 if st.session_state.theme == 'Dark' else 1)
        if theme_selection != st.session_state.theme:
            st.session_state.theme = theme_selection
            st.rerun()
        
        st.write("### Language")
        lang_selection = st.selectbox("Language", ["English", "Thai"], index=0 if st.session_state.lang == 'English' else 1)
        if lang_selection != st.session_state.lang:
            st.session_state.lang = lang_selection
            # Future: Implement translation logic
            st.rerun()
        
        st.write("### AI Configuration")
        api_key = None
        if "GOOGLE_API_KEY" in st.secrets:
            api_key = st.secrets["GOOGLE_API_KEY"]
            st.success("API Key loaded from secrets.")
        else:
            api_key = st.text_input("Gemini API Key", type="password", help="Get your key at aistudio.google.com")
        
        if api_key:
            genai.configure(api_key=api_key)

current_theme = THEME_DARK if st.session_state.theme == 'Dark' else THEME_LIGHT

# --- Custom CSS ---
st.markdown(f"""
    <style>
    /* Hide Streamlit defaults */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    /* header {{visibility: hidden;}}  <-- Commented out to allow sidebar toggle */
    
    /* Global Styles */
    .stApp {{
        background-color: {current_theme['bg']};
        color: {current_theme['text_main']};
    }}
    .block-container {{
        padding-top: 1rem;
        padding-bottom: 1rem;
    }}
    
    /* Card Style */
    .metric-card {{
        background-color: {current_theme['card']};
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }}
    .metric-label {{
        font-size: 14px;
        color: {current_theme['text_sec']};
        margin-bottom: 5px;
    }}
    .metric-value {{
        font-size: 24px;
        font-weight: bold;
        color: {current_theme['text_main']};
    }}
    .metric-delta {{
        font-size: 14px;
    }}
    .positive {{ color: {current_theme['accent_green']}; }}
    .negative {{ color: {current_theme['accent_red']}; }}
    
    /* Input Fields Contrast Fix */
    div[data-baseweb="input"] > div {{
        background-color: {current_theme['input_bg']} !important;
        border: 1px solid {current_theme['input_border']} !important;
    }}
    input {{
        color: {current_theme['input_text']} !important;
    }}
    div[data-baseweb="base-input"] {{
        background-color: {current_theme['input_bg']} !important;
    }}
    
    /* Expander Style adjustment for light mode */
    .streamlit-expanderHeader {{
        color: {current_theme['text_main']};
    }}
    
    </style>
""", unsafe_allow_html=True)

# --- Data Persistence ---
def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        return pd.read_csv(PORTFOLIO_FILE)
    else:
        # Initialize dummy portfolio
        dummy_data = [
            {'Ticker': 'AAPL', 'Quantity': 10, 'Average Cost': 150.0},
            {'Ticker': 'MSFT', 'Quantity': 5, 'Average Cost': 280.0},
            {'Ticker': 'TSLA', 'Quantity': 15, 'Average Cost': 200.0},
            {'Ticker': 'GOOGL', 'Quantity': 8, 'Average Cost': 120.0}
        ]
        df = pd.DataFrame(dummy_data)
        save_portfolio(df)
        return df

def save_portfolio(df):
    df.to_csv(PORTFOLIO_FILE, index=False)

# --- Helper Functions ---
@st.cache_data(ttl=300)
def get_stock_data(tickers):
    if not tickers:
        return {}
    
    data = {}
    try:
        # Batch fetch for efficiency
        tickers_str = " ".join(tickers)
        stocks = yf.Tickers(tickers_str)
        
        for ticker in tickers:
            info = stocks.tickers[ticker].fast_info
            history = stocks.tickers[ticker].history(period="1d")
            dividends = stocks.tickers[ticker].dividends
            
            current_price = info.last_price
            prev_close = info.previous_close
            
            # Estimate annual dividend
            div_yield = info.yield_ if hasattr(info, 'yield_') and info.yield_ else 0
            # Fallback calculation if yield is missing
            if not div_yield and not dividends.empty:
                 last_div = dividends.iloc[-1]
                 # Rough estimate: last div * 4 (quarterly)
                 annual_div = last_div * 4
            else:
                 annual_div = current_price * div_yield if div_yield else 0

            data[ticker] = {
                'current_price': current_price,
                'prev_close': prev_close,
                'annual_dividend': annual_div
            }
    except Exception as e:
        st.error(f"Error fetching data: {e}")
    return data

@st.cache_data(ttl=300)
def get_stock_history(ticker, period="6mo"):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        return hist
    except:
        return pd.DataFrame()

# --- Sidebar (Removed) ---
# st.sidebar.title("Portfolio Manager") ... moved to main page

# --- Main Dashboard ---
st.title("🚀 Modern Portfolio Tracker")

# --- Portfolio Manager ---
with st.expander("Manage Portfolio"):
    # Initialize session state for search
    if 'search_ticker' not in st.session_state:
        st.session_state.search_ticker = ""
    if 'stock_info' not in st.session_state:
        st.session_state.stock_info = None

    # Step 1: Search
    col_search, _ = st.columns([3, 1])
    with col_search:
        search_selection = st.selectbox("🔍 Search by Ticker", POPULAR_TICKERS, index=0)
    
    search_input = ""
    if search_selection == "🔍 Custom Search...":
        search_input = st.text_input("Enter Ticker Symbol", placeholder="e.g., PLTR, COIN").upper()
    else:
        search_input = search_selection.split(" - ")[0]

    if search_input and search_input != st.session_state.search_ticker:
        st.session_state.search_ticker = search_input
        try:
            stock = yf.Ticker(search_input)
            info = stock.fast_info
            # Check if valid by accessing a property
            price = info.last_price
            
            # Get detailed info for logo/name
            detailed_info = stock.info 
            
            # Try to get logo from clearbit first, then yfinance
            domain = detailed_info.get('website', '').replace('https://', '').replace('http://', '').replace('www.', '').split('/')[0]
            logo_url = f"https://logo.clearbit.com/{domain}" if domain else detailed_info.get('logo_url', '')
            
            st.session_state.stock_info = {
                'ticker': search_input,
                'price': price,
                'previous_close': info.previous_close,
                'name': detailed_info.get('longName', search_input),
                'logo': logo_url
            }
        except Exception as e:
            st.error(f"Stock not found: {search_input}")
            st.session_state.stock_info = None

    # Step 2 & 3: Preview and Add
    if st.session_state.stock_info:
        info = st.session_state.stock_info
        
        st.markdown("---")
        col_preview1, col_preview2 = st.columns([1, 3])
        
        with col_preview1:
            if info['logo']:
                st.image(info['logo'], width=80)
            else:
                st.markdown(f"# {info['ticker']}")
        
        with col_preview2:
            st.subheader(info['name'])
            
            # Price Header with Change
            change = info['price'] - info['previous_close']
            change_pct = (change / info['previous_close']) * 100
            color_class = "positive" if change >= 0 else "negative"
            sign = "+" if change >= 0 else ""
            
            st.markdown(f"""
                <div style="font-size: 24px; font-weight: bold;">
                    ${info['price']:,.2f} 
                    <span class="metric-delta {color_class}" style="font-size: 18px; margin-left: 10px;">
                        {sign}{change:,.2f} ({sign}{change_pct:.2f}%)
                    </span>
                </div>
            """, unsafe_allow_html=True)

        # All-Time Price History Chart
        st.subheader("All-Time Price History")
        hist_data = get_stock_history(info['ticker'], period="max")
        
        if not hist_data.empty:
            # Determine Trend Color
            start_price = hist_data['Close'].iloc[0]
            end_price = hist_data['Close'].iloc[-1]
            is_positive = end_price >= start_price
            chart_color = current_theme['accent_green'] if is_positive else current_theme['accent_red']

            fig_line = px.line(
                hist_data, 
                x=hist_data.index, 
                y='Close',
                template='plotly_dark' if st.session_state.theme == 'Dark' else 'plotly_white'
            )
            
            fig_line.update_layout(
                height=350,
                margin=dict(l=0, r=0, t=20, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, title="Date"),
                yaxis=dict(showgrid=False, title="Price"),
                hovermode="x unified"
            )
            
            fig_line.update_traces(
                line_color=chart_color,
                line_width=2
            )
            st.plotly_chart(fig_line, use_container_width=True)
        
        st.markdown("---")
        
        with st.form("add_asset_form"):
            st.write(f"Add **{info['ticker']}** to Portfolio")
            col_inp1, col_inp2 = st.columns(2)
            with col_inp1:
                quantity_input = st.number_input("Quantity", min_value=0.0, step=0.01, value=None, placeholder="0.00")
            with col_inp2:
                cost_input = st.number_input("Average Cost ($)", min_value=0.0, step=0.01, value=None, placeholder="0.00")
            
            add_btn = st.form_submit_button("Add to Portfolio")
            
            if add_btn:
                if quantity_input and quantity_input > 0:
                    df = load_portfolio()
                    ticker = info['ticker']
                    
                    # Upsert logic
                    if ticker in df['Ticker'].values:
                        df.loc[df['Ticker'] == ticker, 'Quantity'] = quantity_input
                        if cost_input:
                            df.loc[df['Ticker'] == ticker, 'Average Cost'] = cost_input
                        st.success(f"Updated {ticker}")
                    else:
                        cost = cost_input if cost_input else info['price'] # Default to current price if not provided
                        new_row = pd.DataFrame([{'Ticker': ticker, 'Quantity': quantity_input, 'Average Cost': cost}])
                        df = pd.concat([df, new_row], ignore_index=True)
                        st.success(f"Added {ticker}")
                    
                    save_portfolio(df)
                    # Clear search after adding
                    st.session_state.stock_info = None
                    st.session_state.search_ticker = "" # Reset search state
                    st.rerun()
                else:
                    st.warning("Please enter a valid quantity.")

    # Delete Section (Separate to keep main flow clean)
    with st.expander("Remove Asset"):
        del_ticker = st.text_input("Ticker to Remove").upper()
        if st.button("Delete Asset"):
            if del_ticker:
                df = load_portfolio()
                if del_ticker in df['Ticker'].values:
                    df = df[df['Ticker'] != del_ticker]
                    save_portfolio(df)
                    st.success(f"Deleted {del_ticker}")
                    st.rerun()
                else:
                    st.warning("Ticker not found.")

    if st.button("Clear All Data"):
        save_portfolio(pd.DataFrame(columns=['Ticker', 'Quantity', 'Average Cost']))
        st.rerun()

df = load_portfolio()

if not df.empty:
    # Fetch Data
    tickers = df['Ticker'].tolist()
    market_data = get_stock_data(tickers)
    
    # Calculate Metrics
    portfolio_data = []
    total_value = 0
    total_cost = 0
    total_daily_change = 0
    total_annual_dividend = 0
    
    for index, row in df.iterrows():
        ticker = row['Ticker']
        qty = row['Quantity']
        avg_cost = row['Average Cost']
        
        if ticker in market_data:
            data = market_data[ticker]
            price = data['current_price']
            prev_close = data['prev_close']
            annual_div = data['annual_dividend']
            
            market_val = qty * price
            cost_basis = qty * avg_cost
            gain_loss = market_val - cost_basis
            gain_loss_pct = (gain_loss / cost_basis) * 100 if cost_basis > 0 else 0
            daily_change = (price - prev_close) * qty
            est_income = qty * annual_div
            
            total_value += market_val
            total_cost += cost_basis
            total_daily_change += daily_change
            total_annual_dividend += est_income
            
            portfolio_data.append({
                'Ticker': ticker,
                'Price': price,
                'Cost Basis': avg_cost,
                'Quantity': qty,
                'Market Value': market_val,
                'Gain/Loss': gain_loss,
                'Gain/Loss %': gain_loss_pct,
                'Daily Change': daily_change,
                'Est. Income': est_income
            })
            
    portfolio_df = pd.DataFrame(portfolio_data)
    
    def color_profit_loss(val):
        color = current_theme['accent_green'] if val >= 0 else current_theme['accent_red']
        return f'color: {color}'

    styled_df = portfolio_df.style.format({
        'Price': '${:.2f}',
        'Cost Basis': '${:.2f}',
        'Market Value': '${:.2f}',
        'Gain/Loss': '${:.2f}',
        'Gain/Loss %': '{:.2f}%',
        'Daily Change': '${:.2f}',
        'Est. Income': '${:.2f}'
    }).map(color_profit_loss, subset=['Gain/Loss', 'Gain/Loss %', 'Daily Change'])

    # Top Metrics Row
    total_gain_loss = total_value - total_cost
    total_gain_loss_pct = (total_gain_loss / total_cost) * 100 if total_cost > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    def metric_card(label, value, delta=None, prefix="$", suffix=""):
        delta_html = ""
        if delta is not None:
            color_class = "positive" if delta >= 0 else "negative"
            sign = "+" if delta >= 0 else ""
            delta_html = f'<div class="metric-delta {color_class}">{sign}{delta:,.2f}{suffix}</div>'
            
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{prefix}{value:,.2f}</div>
                {delta_html}
            </div>
        """, unsafe_allow_html=True)

    with col1:
        metric_card("Total Value", total_value)
    with col2:
        metric_card("Total Gain/Loss", total_gain_loss, total_gain_loss_pct, suffix="%")
    with col3:
        metric_card("Daily Change", total_daily_change)
    with col4:
        metric_card("Est. Annual Income", total_annual_dividend)

    st.markdown("---")

    

 
    # Charts Row
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Asset Allocation")
        fig_donut = px.pie(
            portfolio_df, 
            values='Market Value', 
            names='Ticker', 
            hole=0.6,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_donut.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=current_theme['text_main']),
            showlegend=True,
            margin=dict(t=0, b=0, l=0, r=0)
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        
    with col_chart2:
        st.subheader("Monthly Est. Income")
        # Simple estimation: distribute annual income evenly for now (can be improved with ex-div dates)
        monthly_income = total_annual_dividend / 12
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        income_data = pd.DataFrame({'Month': months, 'Income': [monthly_income]*12})
        
        fig_bar = px.bar(
            income_data,
            x='Month',
            y='Income',
            color_discrete_sequence=[current_theme['accent_green']]
        )
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=current_theme['text_main']),
            margin=dict(t=0, b=0, l=0, r=0),
            yaxis=dict(showgrid=True, gridcolor='#333' if st.session_state.theme == 'Dark' else '#ddd')
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Holdings Table
    st.subheader("Holdings")
    
    st.data_editor(
        styled_df,
        use_container_width=True,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", help="Stock Symbol"),
            "Quantity": st.column_config.NumberColumn("Qty"),
            "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
            "Cost Basis": st.column_config.NumberColumn("Cost Basis", format="$%.2f"),
            "Market Value": st.column_config.NumberColumn("Market Value", format="$%.2f"),
            "Gain/Loss": st.column_config.NumberColumn("Gain/Loss", format="$%.2f"),
            "Gain/Loss %": st.column_config.NumberColumn("Gain/Loss %", format="%.2f%%"),
            "Daily Change": st.column_config.NumberColumn("Daily Change", format="$%.2f"),
            "Est. Income": st.column_config.NumberColumn("Est. Income", format="$%.2f"),
        },
        hide_index=True,
        disabled=True
    )

    # --- AI Portfolio Analyst ---
    st.markdown("---")
    st.subheader("🤖 AI Portfolio Analyst")
    
    with st.expander("Analyze Portfolio with Gemini", expanded=False):
        if not api_key:
            st.warning("Please enter your Gemini API Key in the Settings sidebar to use this feature.")
        else:
            if st.button("Generate Analysis"):
                with st.spinner("Analyzing your portfolio..."):
                    try:
                        # Construct Prompt
                        portfolio_summary = portfolio_df.to_string(index=False)
                        prompt = f"""
                        You are a professional financial advisor. Analyze this stock portfolio:
                        
                        {portfolio_summary}
                        
                        Total Value: ${total_value:,.2f}
                        Total Gain/Loss: ${total_gain_loss:,.2f} ({total_gain_loss_pct:.2f}%)
                        
                        Please provide:
                        1. A brief performance summary.
                        2. An assessment of diversification and risk.
                        3. Three specific, actionable suggestions for improvement.
                        
                        Keep the tone professional but accessible. Use markdown for formatting.
                        """
                        
                        model = genai.GenerativeModel('gemini-pro')
                        response = model.generate_content(prompt)
                        st.markdown(response.text)
                        
                    except Exception as e:
                        st.error(f"AI Analysis Failed: {e}")

else:
    st.info("Your portfolio is empty. Add assets using the 'Manage Portfolio' section above to get started!")
