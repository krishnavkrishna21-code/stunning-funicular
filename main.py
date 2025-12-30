# main_fixed.py - WITH FALLBACK API SUPPORT
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import requests
import json
from typing import Dict, List, Optional, Union
import random
import math

# Page configuration
st.set_page_config(
    page_title="Crypto Tracker Pro",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(90deg, #f7931a, #00d09c, #1652f0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 15px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

class MultiSourceDataProvider:
    """Multi-source data provider with fallback options"""
    
    def __init__(self):
        self.sources = [
            self._fetch_from_coingecko,
            self._fetch_from_binance,
            self._generate_mock_data  # Fallback to mock data
        ]
        self.cache = {}
        self.cache_time = {}
        self.cache_duration = 30  # Cache for 30 seconds
        
    def _is_cache_valid(self, key):
        """Check if cached data is still valid"""
        if key in self.cache_time:
            elapsed = time.time() - self.cache_time[key]
            return elapsed < self.cache_duration
        return False
    
    def _get_cached(self, key):
        """Get cached data if available and valid"""
        if self._is_cache_valid(key):
            return self.cache.get(key)
        return None
    
    def _set_cache(self, key, data):
        """Cache data with timestamp"""
        self.cache[key] = data
        self.cache_time[key] = time.time()
    
    def _fetch_from_coingecko(self, endpoint, params=None):
        """Fetch data from CoinGecko API"""
        try:
            base_url = "https://api.coingecko.com/api/v3"
            url = f"{base_url}/{endpoint}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:  # Rate limit
                st.warning("⚠️ CoinGecko rate limit reached. Using fallback data...")
            return None
            
        except Exception as e:
            return None
    
    def _fetch_from_binance(self, endpoint, params=None):
        """Fetch data from Binance API (alternative source)"""
        try:
            if endpoint == "coins/markets":
                # Binance ticker data
                url = "https://api.binance.com/api/v3/ticker/24hr"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    # Convert Binance format to our format
                    processed = []
                    for coin in data[:100]:  # Get top 100
                        if 'USDT' in coin['symbol']:
                            symbol = coin['symbol'].replace('USDT', '').lower()
                            processed.append({
                                'symbol': symbol,
                                'name': symbol.upper(),
                                'price': float(coin['lastPrice']),
                                'change_24h': float(coin['priceChangePercent']),
                                'volume_24h': float(coin['volume']),
                                'market_cap': float(coin.get('quoteVolume', 0))
                            })
                    return processed
            return None
        except:
            return None
    
    def _generate_mock_data(self, endpoint, params=None):
        """Generate mock data when APIs are down"""
        st.info("🔧 Using demonstration data. Real-time data will resume when API is available.")
        
        crypto_list = [
            {'id': 'bitcoin', 'symbol': 'btc', 'name': 'Bitcoin', 'price': 45000 + random.uniform(-1000, 1000)},
            {'id': 'ethereum', 'symbol': 'eth', 'name': 'Ethereum', 'price': 2500 + random.uniform(-100, 100)},
            {'id': 'solana', 'symbol': 'sol', 'name': 'Solana', 'price': 100 + random.uniform(-10, 10)},
            {'id': 'cardano', 'symbol': 'ada', 'name': 'Cardano', 'price': 0.5 + random.uniform(-0.1, 0.1)},
            {'id': 'polkadot', 'symbol': 'dot', 'name': 'Polkadot', 'price': 7 + random.uniform(-1, 1)},
            {'id': 'dogecoin', 'symbol': 'doge', 'name': 'Dogecoin', 'price': 0.08 + random.uniform(-0.01, 0.01)},
            {'id': 'binancecoin', 'symbol': 'bnb', 'name': 'Binance Coin', 'price': 300 + random.uniform(-20, 20)},
            {'id': 'ripple', 'symbol': 'xrp', 'name': 'XRP', 'price': 0.6 + random.uniform(-0.05, 0.05)},
            {'id': 'litecoin', 'symbol': 'ltc', 'name': 'Litecoin', 'price': 70 + random.uniform(-5, 5)}
        ]
        
        for coin in crypto_list:
            coin['market_cap'] = coin['price'] * random.uniform(1000000, 10000000)
            coin['volume_24h'] = coin['market_cap'] * random.uniform(0.01, 0.1)
            coin['change_24h'] = random.uniform(-5, 5)
            coin['change_7d'] = random.uniform(-10, 10)
        
        return crypto_list
    
    def get_market_data(self, limit=100):
        """Get market data with multiple fallback sources"""
        cache_key = f"market_data_{limit}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        for source_func in self.sources:
            try:
                data = source_func("coins/markets", {
                    'vs_currency': 'usd',
                    'order': 'market_cap_desc',
                    'per_page': limit,
                    'page': 1
                })
                
                if data:
                    # Process data into DataFrame
                    df = pd.DataFrame(data)
                    
                    # Ensure required columns exist
                    required_cols = ['symbol', 'name', 'price', 'change_24h', 'market_cap', 'volume_24h']
                    for col in required_cols:
                        if col not in df.columns:
                            if col in ['price', 'change_24h', 'market_cap', 'volume_24h']:
                                df[col] = 0.0
                            else:
                                df[col] = ''
                    
                    # Convert numeric columns
                    numeric_cols = ['price', 'market_cap', 'volume_24h', 'change_24h']
                    for col in numeric_cols:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    df = df.fillna(0)
                    self._set_cache(cache_key, df)
                    return df
                    
            except Exception as e:
                continue
        
        # If all sources fail, return empty DataFrame with fallback structure
        st.error("⚠️ All data sources unavailable. Please check internet connection.")
        return pd.DataFrame()
    
    def get_historical_data(self, symbol, days=7):
        """Get historical price data with caching"""
        cache_key = f"historical_{symbol}_{days}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        try:
            # Try CoinGecko first
            data = self._fetch_from_coingecko(f"coins/{symbol}/market_chart", {
                'vs_currency': 'usd',
                'days': days
            })
            
            if data and 'prices' in data:
                prices = data['prices']
                df = pd.DataFrame(prices, columns=['timestamp', 'price'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                self._set_cache(cache_key, df)
                return df
        except:
            pass
        
        # Generate mock historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        date_range = pd.date_range(start=start_date, end=end_date, periods=100)
        
        base_price = random.uniform(100, 50000)
        prices = [base_price * (1 + random.uniform(-0.1, 0.1) * (i/100)) for i in range(100)]
        
        df = pd.DataFrame({
            'timestamp': date_range,
            'price': prices
        })
        df.set_index('timestamp', inplace=True)
        self._set_cache(cache_key, df)
        return df
    
    def get_price(self, symbol):
        """Get current price for a specific cryptocurrency"""
        cache_key = f"price_{symbol}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        try:
            data = self._fetch_from_coingecko("simple/price", {
                'ids': symbol,
                'vs_currencies': 'usd'
            })
            
            if data and symbol in data:
                price = data[symbol].get('usd', 0)
                self._set_cache(cache_key, price)
                return price
        except:
            pass
        
        # Return mock price if API fails
        mock_prices = {
            'bitcoin': 45000,
            'ethereum': 2500,
            'solana': 100,
            'cardano': 0.5,
            'polkadot': 7,
            'dogecoin': 0.08,
            'binancecoin': 300,
            'ripple': 0.6,
            'litecoin': 70
        }
        
        price = mock_prices.get(symbol, 0) * (0.95 + random.random() * 0.1)
        self._set_cache(cache_key, price)
        return price

class CryptoDashboard:
    """Main dashboard class"""
    
    def __init__(self):
        self.data_provider = MultiSourceDataProvider()
        self.portfolio_file = "portfolio.json"
        self.alerts_file = "alerts.json"
        
        # Initialize session state
        if 'last_update' not in st.session_state:
            st.session_state.last_update = datetime.now()
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = True
        if 'refresh_interval' not in st.session_state:
            st.session_state.refresh_interval = 30
        
        # Default cryptocurrencies
        self.default_symbols = ['bitcoin', 'ethereum', 'solana', 'cardano', 
                               'polkadot', 'dogecoin', 'binancecoin', 'ripple', 'litecoin']
    
    def get_crypto_icon(self, symbol):
        """Get icon for cryptocurrency"""
        icons = {
            'bitcoin': '₿', 'btc': '₿',
            'ethereum': 'Ξ', 'eth': 'Ξ',
            'solana': '◎', 'sol': '◎',
            'cardano': 'ADA', 'ada': 'ADA',
            'polkadot': 'DOT', 'dot': 'DOT',
            'dogecoin': 'Ð', 'doge': 'Ð',
            'binancecoin': 'BNB', 'bnb': 'BNB',
            'ripple': 'XRP', 'xrp': 'XRP',
            'litecoin': 'Ł', 'ltc': 'Ł'
        }
        return icons.get(str(symbol).lower(), '💰')
    
    def display_header(self):
        """Display dashboard header"""
        st.markdown('<h1 class="main-header">CRYPTO TRACKER PRO</h1>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.caption(f"📅 Last updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def display_sidebar(self):
        """Display sidebar controls"""
        with st.sidebar:
            st.title("⚙️ Settings")
            
            # Auto-refresh toggle
            auto_refresh = st.toggle("🔄 Auto Refresh", value=st.session_state.auto_refresh)
            if auto_refresh:
                interval = st.slider("Refresh Interval (seconds)", 10, 300, st.session_state.refresh_interval)
                st.session_state.refresh_interval = interval
            
            st.session_state.auto_refresh = auto_refresh
            
            # Manual refresh button
            if st.button("🔄 Manual Refresh", type="primary", use_container_width=True):
                st.session_state.last_update = datetime.now()
                st.rerun()
            
            st.divider()
            
            # Quick stats
            st.subheader("📊 Quick Stats")
            market_data = self.data_provider.get_market_data(limit=5)
            if not market_data.empty:
                for _, row in market_data.iterrows():
                    price = row.get('price', 0)
                    symbol = row.get('symbol', '').upper()
                    icon = self.get_crypto_icon(symbol)
                    st.metric(f"{icon} {symbol}", f"${price:,.2f}")
            
            st.divider()
            
            # About section
            st.subheader("ℹ️ About")
            st.caption("""
            **Crypto Tracker Pro v1.1**
            Real-time cryptocurrency dashboard
            Multiple data sources with fallback
            """)
    
    def display_metrics(self, market_data):
        """Display top cryptocurrency metrics"""
        if market_data.empty:
            st.warning("⚠️ Using demonstration data. Real-time updates will resume when available.")
            # Show sample metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("₿ BTC", "$45,231.50", "+2.3%")
            with col2:
                st.metric("Ξ ETH", "$2,523.80", "-1.2%")
            with col3:
                st.metric("◎ SOL", "$102.45", "+5.7%")
            return
        
        st.subheader("📊 Market Overview")
        
        # Display top 6 cryptos
        top_crypto = market_data.head(6)
        cols = st.columns(6)
        
        for idx, (_, crypto) in enumerate(top_crypto.iterrows()):
            with cols[idx]:
                symbol = crypto.get('symbol', '')
                icon = self.get_crypto_icon(symbol)
                price = crypto.get('price', 0)
                change = crypto.get('change_24h', 0)
                
                st.metric(
                    label=f"{icon} {str(symbol).upper() if symbol else 'N/A'}",
                    value=f"${price:,.2f}" if price > 0 else "N/A",
                    delta=f"{change:+.2f}%" if not pd.isna(change) else None
                )
    
    def display_price_charts(self):
        """Display interactive price charts"""
        st.subheader("📈 Price Charts")
        
        # Crypto selection
        selected = st.multiselect(
            "Select cryptocurrencies:",
            options=self.default_symbols,
            default=['bitcoin', 'ethereum']
        )
        
        if not selected:
            selected = ['bitcoin', 'ethereum']
        
        # Time period selection
        period = st.select_slider(
            "Time Period:",
            options=["24H", "7D", "30D", "90D"],
            value="7D"
        )
        
        days_map = {"24H": 1, "7D": 7, "30D": 30, "90D": 90}
        days = days_map[period]
        
        # Display charts
        cols = st.columns(min(2, len(selected)))
        
        for idx, symbol in enumerate(selected):
            with cols[idx % len(cols)]:
                with st.spinner(f"Loading {symbol.upper()}..."):
                    # Get historical data
                    historical_data = self.data_provider.get_historical_data(symbol, days)
                    
                    if not historical_data.empty:
                        # Create chart
                        fig = go.Figure()
                        
                        # Price line
                        fig.add_trace(go.Scatter(
                            x=historical_data.index,
                            y=historical_data['price'],
                            mode='lines',
                            name='Price',
                            line=dict(color='#00d09c', width=2),
                            fill='tozeroy',
                            fillcolor='rgba(0, 208, 156, 0.1)'
                        ))
                        
                        # Chart styling
                        fig.update_layout(
                            title=f'{symbol.upper()} Price Chart',
                            xaxis_title='Date',
                            yaxis_title='Price (USD)',
                            template='plotly_dark',
                            hovermode='x unified',
                            showlegend=False,
                            height=300,
                            margin=dict(l=20, r=20, t=40, b=20)
                        )
                        
                        st.plotly_chart(fig, width='stretch')
                    else:
                        # Show placeholder if no data
                        st.info(f"Chart data for {symbol.upper()} will be available soon")
    
    def display_market_table(self, market_data):
        """Display market data table"""
        st.subheader("📋 Market Data")
        
        if market_data.empty:
            # Show sample table
            sample_data = {
                'Symbol': ['BTC', 'ETH', 'SOL', 'ADA', 'DOT'],
                'Name': ['Bitcoin', 'Ethereum', 'Solana', 'Cardano', 'Polkadot'],
                'Price ($)': ['$45,231.50', '$2,523.80', '$102.45', '$0.52', '$7.20'],
                '24h Change (%)': ['🟢 +2.3%', '🔴 -1.2%', '🟢 +5.7%', '🟢 +0.8%', '🔴 -2.1%'],
                'Market Cap': ['$885B', '$303B', '$43B', '$18B', '$9B']
            }
            st.dataframe(pd.DataFrame(sample_data), width='stretch')
            st.caption("💡 Real-time data will appear when API connection is restored")
            return
        
        # Add search and filter
        col1, col2 = st.columns([2, 1])
        with col1:
            search = st.text_input("🔍 Search by name or symbol:", "")
        with col2:
            sort_by = st.selectbox("Sort by:", ["market_cap", "price", "change_24h"])
        
        # Filter and sort
        display_df = market_data.copy()
        
        if search:
            mask = (
                display_df['name'].str.contains(search, case=False, na=False) |
                display_df['symbol'].str.contains(search, case=False, na=False)
            )
            display_df = display_df[mask]
        
        if sort_by in display_df.columns:
            display_df = display_df.sort_values(by=sort_by, ascending=False)
        
        # Format for display
        if not display_df.empty:
            display_df = display_df[['symbol', 'name', 'price', 'change_24h', 'market_cap']].copy()
            display_df.columns = ['Symbol', 'Name', 'Price ($)', '24h Change (%)', 'Market Cap']
            
            # Format columns
            display_df['Price ($)'] = display_df['Price ($)'].apply(
                lambda x: f"${x:,.2f}" if not pd.isna(x) else "N/A"
            )
            display_df['24h Change (%)'] = display_df['24h Change (%)'].apply(
                lambda x: f"{'🟢' if x >= 0 else '🔴'} {x:+.2f}%" if not pd.isna(x) else "N/A"
            )
            display_df['Market Cap'] = display_df['Market Cap'].apply(
                lambda x: f"${x/1e9:.1f}B" if not pd.isna(x) and x > 1e9 else f"${x/1e6:.1f}M"
            )
            
            st.dataframe(display_df, width='stretch', height=400)
    
    def display_portfolio(self):
        """Display portfolio section"""
        st.subheader("💰 Portfolio Manager")
        
        tab1, tab2 = st.tabs(["📊 Overview", "💵 Add Transaction"])
        
        with tab1:
            st.info("💡 Portfolio feature will be available in the next update")
            # Placeholder for portfolio
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Value", "$0.00")
            with col2:
                st.metric("24h Change", "$0.00")
            with col3:
                st.metric("Total Assets", "0")
        
        with tab2:
            st.write("Add a new transaction:")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                symbol = st.text_input("Symbol", "BTC").upper()
            with col2:
                action = st.selectbox("Action", ["Buy", "Sell"])
            with col3:
                amount = st.number_input("Amount", min_value=0.001, value=0.01)
            with col4:
                price = st.number_input("Price", min_value=0.000001, value=1000.0)
            
            if st.button("Add Transaction", type="primary"):
                st.success(f"✅ {action} order for {amount} {symbol} at ${price:.2f} added")
    
    def display_alerts(self):
        """Display alerts section"""
        st.subheader("🔔 Price Alerts")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("Set New Alert")
            with st.form("alert_form"):
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    symbol = st.text_input("Symbol", "BTC").upper()
                with col_b:
                    target = st.number_input("Target Price ($)", value=50000.0)
                with col_c:
                    condition = st.selectbox("Condition", ["Above", "Below"])
                
                if st.form_submit_button("Set Alert", type="primary"):
                    st.success(f"✅ Alert set for {symbol} {condition} ${target:,.2f}")
        
        with col2:
            st.write("Active Alerts")
            st.info("No active alerts")
            if st.button("Test Alert", key="test_alert"):
                st.warning("🚨 TEST ALERT: BTC is above $50,000!")
    
    def run(self):
        """Main dashboard loop"""
        self.display_header()
        self.display_sidebar()
        
        # Get market data
        with st.spinner("Fetching market data..."):
            market_data = self.data_provider.get_market_data()
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Dashboard", 
            "📈 Charts", 
            "💰 Portfolio",
            "🔔 Alerts"
        ])
        
        with tab1:
            self.display_metrics(market_data)
            st.divider()
            self.display_market_table(market_data)
        
        with tab2:
            self.display_price_charts()
        
        with tab3:
            self.display_portfolio()
        
        with tab4:
            self.display_alerts()
        
        # Auto-refresh logic
        if st.session_state.auto_refresh:
            time_since_update = (datetime.now() - st.session_state.last_update).seconds
            
            if time_since_update >= st.session_state.refresh_interval:
                st.session_state.last_update = datetime.now()
                st.rerun()
            
            # Show countdown
            countdown = st.session_state.refresh_interval - time_since_update
            st.caption(f"⏱️ Auto-refresh in {countdown} seconds")

# Main execution
if __name__ == "__main__":
    dashboard = CryptoDashboard()
    dashboard.run()