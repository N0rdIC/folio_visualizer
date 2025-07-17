import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import calendar
import json
import io
import time
import random
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configuration
st.set_page_config(
    page_title="Portfolio Optimizer & Analyzer", 
    page_icon="💼", 
    layout="wide"
)

class PortfolioAnalyzer:
    def __init__(self):
        self.openai_api_key = st.secrets.get("OPENAI_API_KEY", None)
        
        # Initialize session state for portfolio persistence
        self.init_session_state()
        
        # Sector mappings for common stocks
        self.sector_mapping = {
            # Technology
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'AMZN': 'Technology',
            'META': 'Technology', 'NVDA': 'Technology', 'ADBE': 'Technology', 'CRM': 'Technology',
            'ORCL': 'Technology', 'IBM': 'Technology', 'INTC': 'Technology', 'AMD': 'Technology',
            
            # Healthcare
            'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare', 'ABBV': 'Healthcare',
            'LLY': 'Healthcare', 'TMO': 'Healthcare', 'ABT': 'Healthcare', 'MRK': 'Healthcare',
            'MDT': 'Healthcare', 'BMY': 'Healthcare', 'GILD': 'Healthcare', 'AMGN': 'Healthcare',
            
            # Financial Services
            'JPM': 'Financial Services', 'BAC': 'Financial Services', 'V': 'Financial Services',
            'WFC': 'Financial Services', 'GS': 'Financial Services', 'MS': 'Financial Services',
            'C': 'Financial Services', 'AXP': 'Financial Services', 'BLK': 'Financial Services',
            
            # Consumer Goods
            'PG': 'Consumer Goods', 'KO': 'Consumer Goods', 'PEP': 'Consumer Goods',
            'WMT': 'Consumer Goods', 'HD': 'Consumer Goods', 'MCD': 'Consumer Goods',
            'NKE': 'Consumer Goods', 'SBUX': 'Consumer Goods', 'TGT': 'Consumer Goods',
            
            # Energy
            'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'EOG': 'Energy',
            'SLB': 'Energy', 'OXY': 'Energy', 'PSX': 'Energy',
            
            # Utilities
            'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities', 'AEP': 'Utilities',
            'EXC': 'Utilities', 'D': 'Utilities', 'PCG': 'Utilities',
            
            # Industrial
            'CAT': 'Industrial', 'BA': 'Industrial', 'GE': 'Industrial', 'MMM': 'Industrial',
            'HON': 'Industrial', 'UNP': 'Industrial', 'LMT': 'Industrial', 'RTX': 'Industrial',
            
            # Real Estate
            'AMT': 'Real Estate', 'PLD': 'Real Estate', 'CCI': 'Real Estate', 'EQIX': 'Real Estate',
            
            # Materials
            'LIN': 'Materials', 'APD': 'Materials', 'ECL': 'Materials', 'FCX': 'Materials',
            
            # Communication Services
            'T': 'Communication Services', 'VZ': 'Communication Services', 'CMCSA': 'Communication Services',
            'DIS': 'Communication Services', 'NFLX': 'Communication Services'
        }

    def init_session_state(self):
        """Initialize session state for portfolio persistence"""
        if 'portfolio_holdings' not in st.session_state:
            st.session_state.portfolio_holdings = []
        if 'last_analysis_date' not in st.session_state:
            st.session_state.last_analysis_date = None
        if 'portfolio_name' not in st.session_state:
            st.session_state.portfolio_name = "My Portfolio"
        if 'portfolio_notes' not in st.session_state:
            st.session_state.portfolio_notes = ""
        if 'optimization_mode' not in st.session_state:
            st.session_state.optimization_mode = False

    def save_portfolio_to_session(self, holdings: List[Dict], name: str = None, notes: str = ""):
        """Save portfolio to session state"""
        st.session_state.portfolio_holdings = holdings.copy()
        st.session_state.last_analysis_date = datetime.now().isoformat()
        if name:
            st.session_state.portfolio_name = name
        st.session_state.portfolio_notes = notes

    def load_portfolio_from_session(self) -> List[Dict]:
        """Load portfolio from session state"""
        return st.session_state.portfolio_holdings.copy()

    def clear_session_data(self):
        """Clear all session data"""
        st.session_state.portfolio_holdings = []
        st.session_state.last_analysis_date = None
        st.session_state.portfolio_name = "My Portfolio"
        st.session_state.portfolio_notes = ""

    def calculate_synthesis_metrics(self, portfolio_data: pd.DataFrame, metrics: Dict, historical_performance: pd.DataFrame = None) -> Dict:
        """Calculate synthesis metrics for portfolio optimization"""
        synthesis = {}
        
        # 1. Sharpe Ratio (enhanced calculation)
        risk_free_rate = 0.04  # 4% risk-free rate
        if historical_performance is not None and not historical_performance.empty and len(historical_performance) > 10:
            # Calculate from actual historical data
            returns = historical_performance['Total_Return'].pct_change().dropna()
            if len(returns) > 5:
                mean_return = returns.mean() * 252  # Annualized
                volatility = returns.std() * np.sqrt(252)  # Annualized
                synthesis['sharpe_ratio'] = (mean_return - risk_free_rate) / volatility if volatility > 0 else 0
            else:
                # Fallback to estimate
                estimated_return = metrics['portfolio_dividend_yield'] / 100 + 0.08
                estimated_volatility = metrics['portfolio_beta'] * 0.16
                synthesis['sharpe_ratio'] = (estimated_return - risk_free_rate) / estimated_volatility if estimated_volatility > 0 else 0
        else:
            # Estimate based on portfolio metrics
            estimated_return = metrics['portfolio_dividend_yield'] / 100 + 0.08  # Dividend yield + estimated price appreciation
            estimated_volatility = metrics['portfolio_beta'] * 0.16  # Market volatility * portfolio beta
            synthesis['sharpe_ratio'] = (estimated_return - risk_free_rate) / estimated_volatility if estimated_volatility > 0 else 0
        
        # 2. Mean Opportunity Margin (weighted average of profit margins)
        # Using P/E ratio as a proxy for valuation opportunity
        valid_pe_ratios = portfolio_data[portfolio_data['pe_ratio'] > 0]['pe_ratio']
        if len(valid_pe_ratios) > 0:
            market_pe = 20  # Assumed market average P/E
            portfolio_weights = portfolio_data[portfolio_data['pe_ratio'] > 0]['shares'] * portfolio_data[portfolio_data['pe_ratio'] > 0]['current_price']
            portfolio_weights = portfolio_weights / portfolio_weights.sum()
            
            # Calculate opportunity margin as deviation from market PE
            weighted_pe = (valid_pe_ratios * portfolio_weights).sum()
            synthesis['mean_opportunity_margin'] = ((market_pe - weighted_pe) / market_pe) * 100  # Percentage below/above market
        else:
            synthesis['mean_opportunity_margin'] = 0
        
        # 3. Annualized Total Return with Dividends Reinvested
        if historical_performance is not None and not historical_performance.empty and len(historical_performance) > 250:
            # Calculate from actual data
            years = len(historical_performance) / 252  # Approximate trading days per year
            total_return = historical_performance['Total_Return'].iloc[-1] / 100
            synthesis['annualized_total_return'] = ((1 + total_return) ** (1/years) - 1) * 100
        else:
            # Estimate based on current metrics
            dividend_yield = metrics['portfolio_dividend_yield']
            estimated_price_appreciation = 8  # 8% estimated annual price appreciation
            synthesis['annualized_total_return'] = dividend_yield + estimated_price_appreciation
        
        # 4. Sector Diversification Score
        synthesis['sector_diversification_score'] = self.calculate_diversification_score(portfolio_data, metrics)
        
        # 5. Overall Portfolio Score (weighted combination)
        # Normalize metrics to 0-100 scale
        sharpe_score = min(100, max(0, (synthesis['sharpe_ratio'] + 1) * 50))  # Sharpe of 1 = 100 points
        margin_score = min(100, max(0, synthesis['mean_opportunity_margin'] + 50))  # 0% margin = 50 points
        return_score = min(100, max(0, synthesis['annualized_total_return'] * 5))  # 20% return = 100 points
        div_score = synthesis['sector_diversification_score']
        
        # Weighted portfolio score
        synthesis['overall_portfolio_score'] = (
            sharpe_score * 0.3 + 
            margin_score * 0.2 + 
            return_score * 0.3 + 
            div_score * 0.2
        )
        
        return synthesis

    def calculate_diversification_score(self, portfolio_data: pd.DataFrame, metrics: Dict) -> float:
        """Calculate sector diversification score (0-100)"""
        sector_allocation = metrics['sector_allocation']
        
        # Number of sectors (more is better)
        num_sectors = len(sector_allocation)
        sector_points = min(50, num_sectors * 8)  # Max 50 points for sectors
        
        # Concentration penalty (lower concentration is better)
        max_sector_weight = sector_allocation['weight'].max()
        concentration_penalty = max(0, (max_sector_weight - 0.25) * 100)  # Penalty for >25% in one sector
        
        # Balance score (closer to equal distribution is better)
        ideal_weight = 1.0 / num_sectors if num_sectors > 0 else 0
        balance_score = 0
        if num_sectors > 0:
            deviations = abs(sector_allocation['weight'] - ideal_weight)
            balance_score = max(0, 30 - deviations.sum() * 100)  # Max 30 points for balance
        
        # Total score
        total_score = sector_points - concentration_penalty + balance_score
        return max(0, min(100, total_score))

    def add_stock_proportional(self, symbol: str, current_holdings: List[Dict], target_weight: float = None) -> List[Dict]:
        """Add a stock with proportional allocation"""
        if not current_holdings:
            # If no holdings, start with a base amount
            return [{'symbol': symbol, 'shares': 100}]
        
        # Calculate total current value (this would need current prices, so we'll use a simplified approach)
        # For now, assume equal weighting if no target weight specified
        if target_weight is None:
            target_weight = 1.0 / (len(current_holdings) + 1)  # Equal weight among all stocks
        
        # Create new holdings list
        new_holdings = current_holdings.copy()
        
        # Adjust existing holdings to make room for new stock
        adjustment_factor = 1 - target_weight
        for holding in new_holdings:
            holding['shares'] = int(holding['shares'] * adjustment_factor)
        
        # Add new stock (we'll need to fetch price to calculate shares, for now use placeholder)
        new_holdings.append({'symbol': symbol, 'shares': 100})  # Placeholder, will be calculated in analysis
        
        return new_holdings

    # [Keep all existing methods from the original class...]
    # Including: fetch_stock_info, calculate_portfolio_metrics, simulate_historical_performance, etc.
    # [For brevity, I'm showing the new/modified methods above and indicating the rest should remain]

    def fetch_alpha_vantage_data(self, symbol: str) -> Dict:
        """Fetch data from Alpha Vantage API"""
        api_key = st.secrets.get("ALPHA_VANTAGE_API_KEY")
        if not api_key or api_key == "demo":
            return None
            
        try:
            # Get quote data
            quote_url = f"https://www.alphavantage.co/query"
            quote_params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': api_key
            }
            
            response = requests.get(quote_url, params=quote_params, timeout=10)
            data = response.json()
            
            if 'Global Quote' in data:
                quote = data['Global Quote']
                current_price = float(quote.get('05. price', 0))
                
                # Get basic company overview
                overview_params = {
                    'function': 'OVERVIEW',
                    'symbol': symbol,
                    'apikey': api_key
                }
                
                overview_response = requests.get(quote_url, params=overview_params, timeout=10)
                overview_data = overview_response.json()
                
                return {
                    'symbol': symbol,
                    'name': overview_data.get('Name', symbol),
                    'sector': overview_data.get('Sector', 'Unknown'),
                    'industry': overview_data.get('Industry', 'Unknown'),
                    'current_price': current_price,
                    'dividend_yield': float(overview_data.get('DividendYield', 0)) * 100 if overview_data.get('DividendYield') else 0,
                    'dividend_rate': float(overview_data.get('DividendPerShare', 0)),
                    'payout_ratio': 0,  # Not available in Alpha Vantage
                    'pe_ratio': float(overview_data.get('PERatio', 0)),
                    'market_cap': int(overview_data.get('MarketCapitalization', 0)),
                    'beta': float(overview_data.get('Beta', 1)),
                    'ex_dividend_date': overview_data.get('ExDividendDate'),
                    'dividend_date': None,
                    'country': overview_data.get('Country', 'Unknown')
                }
            
        except Exception as e:
            st.warning(f"Alpha Vantage error for {symbol}: {str(e)}")
            return None
        
        return None

    def fetch_fmp_data(self, symbol: str) -> Dict:
        """Fetch data from Financial Modeling Prep API"""
        api_key = st.secrets.get("FMP_API_KEY")
        if not api_key or api_key == "demo":
            return None
            
        try:
            # Get quote data
            quote_url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}"
            quote_params = {'apikey': api_key}
            
            response = requests.get(quote_url, params=quote_params, timeout=10)
            data = response.json()
            
            if data and len(data) > 0:
                quote = data[0]
                
                # Get profile data
                profile_url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}"
                profile_response = requests.get(profile_url, params=quote_params, timeout=10)
                profile_data = profile_response.json()
                
                profile = profile_data[0] if profile_data else {}
                
                return {
                    'symbol': symbol,
                    'name': profile.get('companyName', symbol),
                    'sector': profile.get('sector', 'Unknown'),
                    'industry': profile.get('industry', 'Unknown'),
                    'current_price': quote.get('price', 0),
                    'dividend_yield': (quote.get('price', 1) / profile.get('lastDiv', 0) * 100) if profile.get('lastDiv') else 0,
                    'dividend_rate': profile.get('lastDiv', 0),
                    'payout_ratio': 0,  # Would need additional API call
                    'pe_ratio': quote.get('pe', 0),
                    'market_cap': profile.get('mktCap', 0),
                    'beta': profile.get('beta', 1),
                    'ex_dividend_date': None,
                    'dividend_date': None,
                    'country': profile.get('country', 'Unknown')
                }
            
        except Exception as e:
            st.warning(f"FMP error for {symbol}: {str(e)}")
            return None
        
        return None

    def manual_data_entry_fallback(self, symbol: str) -> Dict:
        """Allow manual data entry when APIs fail"""
        st.warning(f"⚠️ All APIs failed for {symbol}. Using fallback values.")
        
        # Return basic structure with default values
        return {
            'symbol': symbol,
            'name': symbol,
            'sector': self.sector_mapping.get(symbol, 'Unknown'),
            'industry': 'Unknown',
            'current_price': 0,  # User would need to enter manually
            'dividend_yield': 0,
            'dividend_rate': 0,
            'payout_ratio': 0,
            'pe_ratio': 0,
            'market_cap': 0,
            'beta': 1,
            'ex_dividend_date': None,
            'dividend_date': None,
            'country': 'Unknown'
        }

    def fetch_stock_info_fallback(self, symbol: str) -> Dict:
        """Fallback method using alternative data sources when Yahoo Finance is blocked"""
        
        # Try Alpha Vantage if API key is available
        if hasattr(st.secrets, "ALPHA_VANTAGE_API_KEY") and st.secrets.ALPHA_VANTAGE_API_KEY != "demo":
            try:
                av_data = self.fetch_alpha_vantage_data(symbol)
                if av_data and av_data['current_price'] > 0:
                    return av_data
            except Exception as e:
                st.warning(f"Alpha Vantage fallback failed for {symbol}: {str(e)}")
        
        # Try Financial Modeling Prep if API key is available  
        if hasattr(st.secrets, "FMP_API_KEY") and st.secrets.FMP_API_KEY != "demo":
            try:
                fmp_data = self.fetch_fmp_data(symbol)
                if fmp_data and fmp_data['current_price'] > 0:
                    return fmp_data
            except Exception as e:
                st.warning(f"FMP fallback failed for {symbol}: {str(e)}")
        
        # Manual data entry fallback
        return self.manual_data_entry_fallback(symbol)

    def fetch_stock_info(self, symbol: str, retry_count: int = 3, use_fallback: bool = True) -> Dict:
        """Fetch comprehensive stock information with rate limiting and fallback options"""
        
        # MANDATORY 1-second delay for Yahoo Finance to avoid rate limiting
        time.sleep(1.0)
        
        # Add additional random delay to avoid hitting rate limits
        additional_delay = random.uniform(0.2, 0.5)
        time.sleep(additional_delay)
        
        for attempt in range(retry_count):
            try:
                stock = yf.Ticker(symbol)
                info = stock.info
                
                # Check if we got valid data
                if not info or len(info) < 5:
                    raise Exception("Empty or minimal data returned")
                
                # Handle dividend yield properly - yfinance returns it as decimal (0.0152 for 1.52%)
                dividend_yield_raw = info.get('dividendYield', 0)
                if dividend_yield_raw:
                    # If the value is already > 1, it might be in percentage form, don't multiply
                    dividend_yield = dividend_yield_raw * 100 if dividend_yield_raw < 1 else dividend_yield_raw
                else:
                    dividend_yield = 0
                
                # Handle payout ratio similarly
                payout_ratio_raw = info.get('payoutRatio', 0)
                if payout_ratio_raw:
                    payout_ratio = payout_ratio_raw * 100 if payout_ratio_raw < 1 else payout_ratio_raw
                else:
                    payout_ratio = 0
                
                return {
                    'symbol': symbol,
                    'name': info.get('longName', symbol),
                    'sector': info.get('sector', self.sector_mapping.get(symbol, 'Unknown')),
                    'industry': info.get('industry', 'Unknown'),
                    'current_price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                    'dividend_yield': dividend_yield,
                    'dividend_rate': info.get('dividendRate', 0),
                    'payout_ratio': payout_ratio,
                    'pe_ratio': info.get('forwardPE', info.get('trailingPE', 0)),
                    'market_cap': info.get('marketCap', 0),
                    'beta': info.get('beta', 1),
                    'ex_dividend_date': info.get('exDividendDate'),
                    'dividend_date': info.get('dividendDate'),
                    'country': info.get('country', 'Unknown')
                }
                
            except Exception as e:
                error_msg = str(e).lower()
                if "rate limit" in error_msg or "too many requests" in error_msg or "403" in error_msg or "blocked" in error_msg:
                    # Exponential backoff for rate limiting with longer delays
                    wait_time = 2 + (2 ** attempt) + random.uniform(1, 3)
                    st.warning(f"⚠️ Yahoo Finance rate limited for {symbol}. Waiting {wait_time:.1f}s before retry {attempt + 1}/{retry_count}...")
                    time.sleep(wait_time)
                    continue
                else:
                    st.warning(f"Yahoo Finance error for {symbol}: {str(e)}")
                    break
        
        # If all Yahoo Finance attempts failed, try fallback sources
        if use_fallback:
            st.info(f"🔄 Trying alternative data sources for {symbol}...")
            fallback_data = self.fetch_stock_info_fallback(symbol)
            if fallback_data and fallback_data['current_price'] > 0:
                return fallback_data
        
        # Return default values if all attempts failed
        st.error(f"❌ All data sources failed for {symbol}. Using default values.")
        return {
            'symbol': symbol,
            'name': symbol,
            'sector': self.sector_mapping.get(symbol, 'Unknown'),
            'industry': 'Unknown',
            'current_price': 0,
            'dividend_yield': 0,
            'dividend_rate': 0,
            'payout_ratio': 0,
            'pe_ratio': 0,
            'market_cap': 0,
            'beta': 1,
            'ex_dividend_date': None,
            'dividend_date': None,
            'country': 'Unknown'
        }

    def calculate_portfolio_metrics(self, portfolio_data: pd.DataFrame) -> Dict:
        """Calculate comprehensive portfolio metrics including advanced risk measures"""
        metrics = {}
        
        # Current portfolio value
        total_value = (portfolio_data['shares'] * portfolio_data['current_price']).sum()
        metrics['total_value'] = total_value
        
        # Portfolio weights
        portfolio_data['weight'] = (portfolio_data['shares'] * portfolio_data['current_price']) / total_value
        
        # Annual dividend income
        annual_dividends = (portfolio_data['shares'] * portfolio_data['dividend_rate']).sum()
        metrics['annual_dividends'] = annual_dividends
        
        # Portfolio dividend yield
        metrics['portfolio_dividend_yield'] = (annual_dividends / total_value) * 100 if total_value > 0 else 0
        
        # Weighted average P/E ratio
        portfolio_data['pe_weighted'] = portfolio_data['weight'] * portfolio_data['pe_ratio']
        metrics['weighted_pe'] = portfolio_data['pe_weighted'].sum()
        
        # Portfolio beta
        portfolio_data['beta_weighted'] = portfolio_data['weight'] * portfolio_data['beta']
        metrics['portfolio_beta'] = portfolio_data['beta_weighted'].sum()
        
        # Calculate advanced risk metrics
        try:
            # Sharpe Ratio estimation (simplified)
            # Assuming risk-free rate of 4% and estimated portfolio volatility
            risk_free_rate = 0.04
            estimated_return = metrics['portfolio_dividend_yield'] / 100 + 0.08  # Dividend yield + estimated price appreciation
            estimated_volatility = metrics['portfolio_beta'] * 0.16  # Market volatility * portfolio beta
            
            metrics['sharpe_ratio'] = (estimated_return - risk_free_rate) / estimated_volatility if estimated_volatility > 0 else 0
            
            # Alpha estimation (simplified)
            # Alpha = Portfolio Return - (Risk-free Rate + Beta * (Market Return - Risk-free Rate))
            market_return = 0.10  # Assumed market return
            metrics['portfolio_alpha'] = estimated_return - (risk_free_rate + metrics['portfolio_beta'] * (market_return - risk_free_rate))
            
            # Information Ratio (simplified)
            metrics['information_ratio'] = metrics['portfolio_alpha'] / (estimated_volatility * 0.5) if estimated_volatility > 0 else 0
            
        except Exception as e:
            # Default values if calculation fails
            metrics['sharpe_ratio'] = 0
            metrics['portfolio_alpha'] = 0
            metrics['information_ratio'] = 0
        
        # Sector diversification
        sector_allocation = portfolio_data.groupby('sector').agg({
            'weight': 'sum',
            'shares': lambda x: (portfolio_data.loc[x.index, 'shares'] * 
                               portfolio_data.loc[x.index, 'current_price']).sum()
        })
        metrics['sector_allocation'] = sector_allocation
        
        # Top holdings
        portfolio_data_sorted = portfolio_data.sort_values('weight', ascending=False)
        metrics['top_holdings'] = portfolio_data_sorted.head(10)
        
        return metrics

def main():
    st.title("🎯 Portfolio Optimizer & Analyzer")
    st.markdown("**Intelligent portfolio optimization with synthesis metrics and proportional allocation**")
    
    analyzer = PortfolioAnalyzer()
    
    # Sidebar for portfolio management and input
    st.sidebar.header("💾 Portfolio Management")
    
    # Portfolio session info
    if st.session_state.last_analysis_date:
        last_date = datetime.fromisoformat(st.session_state.last_analysis_date)
        st.sidebar.info(f"📅 Last analysis: {last_date.strftime('%Y-%m-%d %H:%M')}")
        
        if st.session_state.portfolio_holdings:
            st.sidebar.success(f"💼 Saved portfolio: {len(st.session_state.portfolio_holdings)} holdings")
    
    # Optimization mode toggle
    st.session_state.optimization_mode = st.sidebar.checkbox(
        "🎯 Optimization Mode", 
        value=st.session_state.optimization_mode,
        help="Enable portfolio optimization features with synthesis table"
    )
    
    # Portfolio management options
    portfolio_action = st.sidebar.radio(
        "Portfolio Action:",
        ["📝 Create/Edit Portfolio", "📂 Load Saved Portfolio", "📁 Import Portfolio File", "🗑️ Clear Session"]
    )
    
    portfolio_holdings = []  # Initialize portfolio_holdings
    
    if portfolio_action == "🗑️ Clear Session":
        if st.sidebar.button("🗑️ Clear All Data", type="secondary"):
            analyzer.clear_session_data()
            st.sidebar.success("Session cleared!")
            st.rerun()
    
    elif portfolio_action == "📂 Load Saved Portfolio":
        saved_holdings = analyzer.load_portfolio_from_session()
        if saved_holdings:
            st.sidebar.subheader("📋 Saved Portfolio")
            st.sidebar.write(f"**Name:** {st.session_state.portfolio_name}")
            if st.session_state.portfolio_notes:
                st.sidebar.write(f"**Notes:** {st.session_state.portfolio_notes}")
            
            # Display saved holdings
            for holding in saved_holdings:
                st.sidebar.write(f"• {holding['symbol']}: {holding['shares']} shares")
            
            if st.sidebar.button("✅ Use Saved Portfolio", type="primary"):
                portfolio_holdings = saved_holdings
                st.session_state.holdings = saved_holdings  # Also update holdings
                st.sidebar.success("Loaded saved portfolio!")
        else:
            st.sidebar.info("No saved portfolio found. Create one first!")
    
    elif portfolio_action == "📝 Create/Edit Portfolio":
        st.sidebar.header("📝 Portfolio Input")
        
        # Portfolio metadata
        portfolio_name = st.sidebar.text_input(
            "Portfolio Name", 
            value=st.session_state.portfolio_name,
            placeholder="e.g., Optimized Dividend Portfolio"
        )
        
        portfolio_notes = st.sidebar.text_area(
            "Portfolio Notes", 
            value=st.session_state.portfolio_notes,
            placeholder="Investment strategy, optimization goals..."
        )
        
        # Update session state
        st.session_state.portfolio_name = portfolio_name
        st.session_state.portfolio_notes = portfolio_notes
        
        # Portfolio input methods
        input_method = st.sidebar.radio(
            "Choose input method:",
            ["Smart Entry", "Manual Entry", "Upload CSV", "Sample Portfolio"]
        )
        
        if input_method == "Smart Entry":
            st.sidebar.subheader("🎯 Smart Portfolio Entry")
            st.sidebar.info("💡 Add stocks with automatic proportional allocation")
            
            # Load existing holdings from session
            if 'holdings' not in st.session_state:
                st.session_state.holdings = analyzer.load_portfolio_from_session()
            
            # Smart add form with proportional allocation
            with st.sidebar.form("smart_add_holding"):
                symbol = st.text_input("Stock Symbol (e.g., AAPL)", "").upper()
                
                allocation_method = st.radio(
                    "Allocation Method:",
                    ["Equal Weight", "Custom Weight", "Specify Shares"]
                )
                
                if allocation_method == "Custom Weight":
                    target_weight = st.number_input("Target Weight (%)", min_value=1.0, max_value=50.0, value=10.0, step=1.0)
                    target_weight = target_weight / 100
                    shares = None
                elif allocation_method == "Specify Shares":
                    shares = st.number_input("Number of Shares", min_value=0.0, value=0.0, step=1.0)
                    target_weight = None
                else:  # Equal Weight
                    target_weight = None
                    shares = None
                
                if st.form_submit_button("➕ Smart Add"):
                    if symbol:
                        if allocation_method == "Specify Shares" and shares > 0:
                            # Traditional manual entry
                            existing_index = None
                            for i, holding in enumerate(st.session_state.holdings):
                                if holding['symbol'] == symbol:
                                    existing_index = i
                                    break
                            
                            if existing_index is not None:
                                st.session_state.holdings[existing_index]['shares'] = shares
                                st.success(f"Updated {symbol}: {shares} shares")
                            else:
                                st.session_state.holdings.append({'symbol': symbol, 'shares': shares})
                                st.success(f"Added {shares} shares of {symbol}")
                        
                        else:
                            # Proportional allocation
                            if target_weight is None:
                                # Equal weight allocation
                                target_weight = 1.0 / (len(st.session_state.holdings) + 1)
                            
                            # Check if symbol already exists
                            existing_index = None
                            for i, holding in enumerate(st.session_state.holdings):
                                if holding['symbol'] == symbol:
                                    existing_index = i
                                    break
                            
                            if existing_index is not None:
                                st.warning(f"{symbol} already exists. Use manual entry to modify shares.")
                            else:
                                # Add with placeholder shares (will be calculated during analysis)
                                st.session_state.holdings.append({
                                    'symbol': symbol, 
                                    'shares': 0,  # Placeholder
                                    'target_weight': target_weight,
                                    'allocation_method': 'proportional'
                                })
                                st.success(f"Added {symbol} with {target_weight*100:.1f}% target allocation")
            
            # Display current holdings with allocation info
            if st.session_state.holdings:
                st.sidebar.subheader("📊 Portfolio Allocation Preview")
                
                total_manual_holdings = sum(1 for h in st.session_state.holdings if h.get('allocation_method') != 'proportional')
                total_proportional_holdings = sum(1 for h in st.session_state.holdings if h.get('allocation_method') == 'proportional')
                
                st.sidebar.write(f"**Manual Holdings:** {total_manual_holdings}")
                st.sidebar.write(f"**Proportional Holdings:** {total_proportional_holdings}")
                
                for holding in st.session_state.holdings:
                    if holding.get('allocation_method') == 'proportional':
                        weight = holding.get('target_weight', 0) * 100
                        st.sidebar.write(f"• {holding['symbol']}: {weight:.1f}% target")
                    else:
                        st.sidebar.write(f"• {holding['symbol']}: {holding['shares']} shares")
        
        elif input_method == "Manual Entry":
            st.sidebar.subheader("Add Holdings")
            
            # Load existing holdings from session
            if 'holdings' not in st.session_state:
                st.session_state.holdings = analyzer.load_portfolio_from_session()
            
            # Add new holding form
            with st.sidebar.form("add_holding"):
                symbol = st.text_input("Stock Symbol (e.g., AAPL)", "").upper()
                shares = st.number_input("Number of Shares", min_value=0.0, value=0.0, step=1.0)
                
                if st.form_submit_button("Add to Portfolio"):
                    if symbol and shares > 0:
                        # Check if symbol already exists
                        existing_index = None
                        for i, holding in enumerate(st.session_state.holdings):
                            if holding['symbol'] == symbol:
                                existing_index = i
                                break
                        
                        if existing_index is not None:
                            # Update existing holding
                            st.session_state.holdings[existing_index]['shares'] = shares
                            st.success(f"Updated {symbol}: {shares} shares")
                        else:
                            # Add new holding
                            st.session_state.holdings.append({'symbol': symbol, 'shares': shares})
                            st.success(f"Added {shares} shares of {symbol}")
            
            # Display current holdings
            if st.session_state.holdings:
                st.sidebar.subheader("📋 Current Holdings")
                for holding in st.session_state.holdings:
                    st.sidebar.write(f"• {holding['symbol']}: {holding['shares']} shares")
        
        elif input_method == "Upload CSV":
            st.sidebar.subheader("Upload Portfolio CSV")
            uploaded_file = st.sidebar.file_uploader(
                "Choose CSV file", 
                type="csv",
                help="CSV should have columns: symbol, shares"
            )
            
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    
                    # Check required columns
                    if 'symbol' not in df.columns:
                        st.sidebar.error("CSV must contain 'symbol' column")
                    elif 'shares' not in df.columns:
                        st.sidebar.error("CSV must contain 'shares' column")
                    else:
                        # Convert to holdings format
                        holdings = []
                        for _, row in df.iterrows():
                            holdings.append({
                                'symbol': str(row['symbol']).upper().strip(),
                                'shares': float(row['shares'])
                            })
                        
                        st.session_state.holdings = holdings
                        st.sidebar.success(f"Loaded {len(holdings)} holdings")
                        
                except Exception as e:
                    st.sidebar.error(f"Error reading CSV: {str(e)}")
        
        else:  # Sample Portfolio
            st.sidebar.subheader("Sample Dividend Portfolio")
            st.session_state.holdings = [
                {'symbol': 'AAPL', 'shares': 50},
                {'symbol': 'MSFT', 'shares': 30},
                {'symbol': 'JNJ', 'shares': 40},
                {'symbol': 'PG', 'shares': 25},
                {'symbol': 'KO', 'shares': 60},
                {'symbol': 'JPM', 'shares': 20},
                {'symbol': 'XOM', 'shares': 35},
                {'symbol': 'VZ', 'shares': 45},
                {'symbol': 'T', 'shares': 55},
                {'symbol': 'PFE', 'shares': 40}
            ]
            st.sidebar.info("Using sample dividend-focused portfolio")
        
        portfolio_holdings = st.session_state.holdings
        
    elif portfolio_action == "📁 Import Portfolio File":
        st.sidebar.subheader("📁 Import Portfolio")
        
        # File upload options
        upload_format = st.sidebar.radio(
            "File Format:",
            ["CSV Format", "JSON Format (Full Backup)"]
        )
        
        uploaded_file = st.sidebar.file_uploader(
            f"Choose {upload_format.split()[0]} file",
            type=["csv"] if "CSV" in upload_format else ["json"],
            help="Upload your portfolio file to restore previous analysis"
        )
        
        if uploaded_file:
            try:
                file_content = uploaded_file.read().decode('utf-8')
                
                if upload_format == "CSV Format":
                    df = pd.read_csv(io.StringIO(file_content))
                    
                    # Check required columns
                    if 'symbol' not in df.columns:
                        st.sidebar.error("CSV must contain 'symbol' column")
                    elif 'shares' not in df.columns:
                        st.sidebar.error("CSV must contain 'shares' column")
                    else:
                        # Convert to holdings format
                        imported_holdings = []
                        for _, row in df.iterrows():
                            imported_holdings.append({
                                'symbol': str(row['symbol']).upper().strip(),
                                'shares': float(row['shares'])
                            })
                        
                        portfolio_holdings = imported_holdings
                        st.session_state.holdings = imported_holdings  # Also update session
                        st.sidebar.success(f"✅ Imported {len(imported_holdings)} holdings from CSV")
                    
                else:  # JSON Format
                    data = json.loads(file_content)
                    
                    # Validate JSON structure
                    if 'holdings' not in data:
                        st.sidebar.error("Invalid portfolio file: missing 'holdings' data")
                    else:
                        imported_holdings = data['holdings']
                        portfolio_holdings = imported_holdings
                        st.session_state.holdings = imported_holdings  # Also update session
                        
                        # Update session with imported metadata
                        st.session_state.portfolio_name = data.get('portfolio_name', 'Imported Portfolio')
                        st.session_state.portfolio_notes = data.get('portfolio_notes', '')
                        
                        st.sidebar.success(f"✅ Imported portfolio: {data.get('portfolio_name', 'Imported Portfolio')}")
                        if data.get('created_date'):
                            st.sidebar.info(f"Created: {data['created_date'][:10]}")
                
            except Exception as e:
                st.sidebar.error(f"❌ Import failed: {str(e)}")
    
    # Main analysis
    if portfolio_holdings and st.sidebar.button("🚀 Analyze Portfolio", type="primary"):
        
        st.info(f"Analyzing portfolio with {len(portfolio_holdings)} holdings...")
        
        # Show optimization mode info
        if st.session_state.optimization_mode:
            st.info("🎯 **Optimization Mode Active** - Enhanced metrics and synthesis table enabled")
        
        # Fetch current data for all holdings
        portfolio_data = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_manual_value = 0
        proportional_holdings = []
        
        # First pass: fetch data for all holdings and calculate manual value
        for i, holding in enumerate(portfolio_holdings):
            symbol = holding['symbol']
            
            status_text.text(f'Fetching data for {symbol}... ({i + 1}/{len(portfolio_holdings)})')
            progress_bar.progress((i + 1) / len(portfolio_holdings))
            
            stock_info = analyzer.fetch_stock_info(symbol)
            
            if holding.get('allocation_method') == 'proportional':
                # Store for second pass
                proportional_holdings.append({
                    'stock_info': stock_info,
                    'target_weight': holding.get('target_weight', 0),
                    'index': i
                })
            else:
                stock_info['shares'] = holding['shares']
                portfolio_data.append(stock_info)
                total_manual_value += stock_info['current_price'] * holding['shares']
        
        # Second pass: calculate proportional shares
        if proportional_holdings:
            # Estimate total portfolio value to calculate proportional shares
            estimated_proportional_value = total_manual_value * 0.3  # Assume proportional holdings are 30% of total
            
            for prop_holding in proportional_holdings:
                stock_info = prop_holding['stock_info']
                target_weight = prop_holding['target_weight']
                
                if stock_info['current_price'] > 0:
                    target_value = estimated_proportional_value * target_weight
                    calculated_shares = target_value / stock_info['current_price']
                    stock_info['shares'] = max(1, int(calculated_shares))  # At least 1 share
                else:
                    stock_info['shares'] = 1  # Default if price unavailable
                
                portfolio_data.append(stock_info)
        
        progress_bar.empty()
        status_text.empty()
        
        portfolio_df = pd.DataFrame(portfolio_data)
        
        # Calculate portfolio metrics
        metrics = analyzer.calculate_portfolio_metrics(portfolio_df)
        
        # Calculate synthesis metrics if in optimization mode
        synthesis_metrics = None
        if st.session_state.optimization_mode:
            synthesis_metrics = analyzer.calculate_synthesis_metrics(portfolio_df, metrics)
        
        # SYNTHESIS TABLE (Top of page in optimization mode)
        if st.session_state.optimization_mode and synthesis_metrics:
            st.header("🎯 Portfolio Optimization Synthesis")
            
            # Create synthesis table
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(
                    "Sharpe Ratio", 
                    f"{synthesis_metrics['sharpe_ratio']:.2f}",
                    help="Risk-adjusted return (>1.0 = good, >2.0 = excellent)"
                )
            
            with col2:
                st.metric(
                    "Opportunity Margin", 
                    f"{synthesis_metrics['mean_opportunity_margin']:.1f}%",
                    help="Valuation opportunity vs market average"
                )
            
            with col3:
                st.metric(
                    "Ann. Total Return", 
                    f"{synthesis_metrics['annualized_total_return']:.1f}%",
                    help="Expected annual return with dividends reinvested"
                )
            
            with col4:
                st.metric(
                    "Diversification Score", 
                    f"{synthesis_metrics['sector_diversification_score']:.0f}/100",
                    help="Sector diversification quality score"
                )
            
            with col5:
                score_color = "🟢" if synthesis_metrics['overall_portfolio_score'] >= 75 else "🟡" if synthesis_metrics['overall_portfolio_score'] >= 50 else "🔴"
                st.metric(
                    "Portfolio Score", 
                    f"{score_color} {synthesis_metrics['overall_portfolio_score']:.0f}/100",
                    help="Overall portfolio optimization score"
                )
            
            # Optimization insights
            st.subheader("💡 Optimization Insights")
            
            insights = []
            
            if synthesis_metrics['sharpe_ratio'] < 0.5:
                insights.append("⚠️ **Low Sharpe Ratio**: Consider adding higher-return or lower-risk assets")
            elif synthesis_metrics['sharpe_ratio'] > 1.5:
                insights.append("✅ **Excellent Sharpe Ratio**: Strong risk-adjusted returns")
            
            if synthesis_metrics['mean_opportunity_margin'] < -20:
                insights.append("📈 **Overvalued Holdings**: Portfolio trading above market multiples")
            elif synthesis_metrics['mean_opportunity_margin'] > 20:
                insights.append("💰 **Value Opportunity**: Portfolio contains undervalued assets")
            
            if synthesis_metrics['sector_diversification_score'] < 50:
                insights.append("🎯 **Improve Diversification**: Add holdings in underrepresented sectors")
            elif synthesis_metrics['sector_diversification_score'] > 80:
                insights.append("✅ **Well Diversified**: Excellent sector distribution")
            
            if synthesis_metrics['annualized_total_return'] < 8:
                insights.append("📊 **Below Market Returns**: Consider higher-growth opportunities")
            elif synthesis_metrics['annualized_total_return'] > 15:
                insights.append("🚀 **Strong Returns**: Excellent total return potential")
            
            if insights:
                for insight in insights:
                    st.info(insight)
            else:
                st.success("🎯 **Well-Optimized Portfolio**: No major optimization opportunities identified")
        
        # Display key metrics
        st.header("📊 Portfolio Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Portfolio Value", 
                f"${metrics['total_value']:,.2f}"
            )
        
        with col2:
            st.metric(
                "Annual Dividend Income", 
                f"${metrics['annual_dividends']:,.2f}",
                f"{metrics['portfolio_dividend_yield']:.2f}% yield"
            )
        
        with col3:
            st.metric(
                "Portfolio Beta", 
                f"{metrics['portfolio_beta']:.2f}",
                "vs Market (1.0)"
            )
        
        with col4:
            st.metric(
                "Weighted P/E Ratio", 
                f"{metrics['weighted_pe']:.1f}",
                "Portfolio Average"
            )
        
        # Holdings table with allocation info
        st.header("📋 Current Holdings")
        
        display_df = portfolio_df.copy()
        display_df['Current Value'] = display_df['shares'] * display_df['current_price']
        display_df['Weight'] = (display_df['Current Value'] / metrics['total_value']) * 100
        display_df['Annual Dividends'] = display_df['shares'] * display_df['dividend_rate']
        
        # Add allocation method info if in optimization mode
        if st.session_state.optimization_mode:
            allocation_info = []
            for _, row in display_df.iterrows():
                symbol = row['symbol']
                # Find original holding info
                orig_holding = next((h for h in portfolio_holdings if h['symbol'] == symbol), None)
                if orig_holding and orig_holding.get('allocation_method') == 'proportional':
                    target_weight = orig_holding.get('target_weight', 0) * 100
                    allocation_info.append(f"Auto ({target_weight:.1f}%)")
                else:
                    allocation_info.append("Manual")
            
            display_df['Allocation'] = allocation_info
        
        # Format for display
        display_columns = {
            'symbol': 'Symbol',
            'name': 'Company',
            'sector': 'Sector',
            'shares': 'Shares',
            'current_price': 'Price',
            'Current Value': 'Current Value',
            'Weight': 'Weight (%)',
            'dividend_yield': 'Div Yield (%)',
            'Annual Dividends': 'Annual Dividends'
        }
        
        if st.session_state.optimization_mode:
            display_columns['Allocation'] = 'Allocation Method'
        
        st.dataframe(
            display_df[list(display_columns.keys())].rename(columns=display_columns),
            column_config={
                "Price": st.column_config.NumberColumn(format="$%.2f"),
                "Current Value": st.column_config.NumberColumn(format="$%.2f"),
                "Weight (%)": st.column_config.NumberColumn(format="%.1f%%"),
                "Div Yield (%)": st.column_config.NumberColumn(format="%.2f%%"),
                "Annual Dividends": st.column_config.NumberColumn(format="$%.2f")
            },
            use_container_width=True
        )
        
        # Visualizations
        st.header("📈 Portfolio Analytics")
        
        # Row 1: Sector allocation and optimization chart
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Sector Allocation")
            sector_data = metrics['sector_allocation'].reset_index()
            
            fig_sector = px.pie(
                sector_data, 
                values='weight', 
                names='sector',
                title="Portfolio Allocation by Sector"
            )
            st.plotly_chart(fig_sector, use_container_width=True)
        
        with col2:
            if st.session_state.optimization_mode and synthesis_metrics:
                st.subheader("🚀 Optimization Radar")
                
                # Create radar chart for optimization metrics
                categories = ['Sharpe Ratio', 'Return Potential', 'Diversification', 'Value Opportunity', 'Overall Score']
                values = [
                    min(100, synthesis_metrics['sharpe_ratio'] * 50),  # Normalize Sharpe
                    min(100, synthesis_metrics['annualized_total_return'] * 5),  # Normalize return
                    synthesis_metrics['sector_diversification_score'],
                    min(100, (synthesis_metrics['mean_opportunity_margin'] + 50)),  # Normalize opportunity
                    synthesis_metrics['overall_portfolio_score']
                ]
                
                fig_radar = go.Figure()
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name='Current Portfolio'
                ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )),
                    showlegend=True,
                    title="Portfolio Optimization Metrics"
                )
                
                st.plotly_chart(fig_radar, use_container_width=True)
            else:
                st.subheader("🏆 Top Holdings")
                top_holdings = metrics['top_holdings'].head(10)
                
                fig_holdings = px.bar(
                    top_holdings,
                    x='weight',
                    y='symbol',
                    orientation='h',
                    title="Top 10 Holdings by Weight",
                    labels={'weight': 'Portfolio Weight', 'symbol': 'Symbol'}
                )
                fig_holdings.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_holdings, use_container_width=True)
        
        # Auto-save analyzed portfolio to session
        analyzer.save_portfolio_to_session(
            portfolio_holdings, 
            st.session_state.portfolio_name, 
            st.session_state.portfolio_notes
        )
        
        st.success("✅ Portfolio analysis complete!")
    
    # Information section
    elif not portfolio_holdings:
        st.info("👈 Please add your portfolio holdings using the sidebar to begin analysis")
        
        # Show session restoration info if available
        if st.session_state.portfolio_holdings:
            st.warning(f"💡 You have a saved portfolio ({len(st.session_state.portfolio_holdings)} holdings). Use 'Load Saved Portfolio' to restore it.")
        
        st.markdown("---")
        st.subheader("🎯 Portfolio Optimization Features")
        st.markdown("""
        **New Optimization Features:**
        
        🎯 **Synthesis Table:**
        - Sharpe coefficient for risk-adjusted returns
        - Mean opportunity margin (valuation analysis)
        - Annualized total return with dividend reinvestment
        - Sector diversification score (0-100)
        - Overall portfolio optimization score
        
        ⚖️ **Smart Proportional Allocation:**
        - Add stocks without specifying shares
        - Automatic equal-weight or custom-weight allocation
        - Portfolio rebalancing suggestions
        - Target allocation tracking
        
        📊 **Enhanced Analytics:**
        - Optimization radar chart
        - Real-time optimization insights
        - Sector concentration analysis
        - Value opportunity identification
        
        🚀 **Iterative Optimization:**
        - Compare multiple portfolio configurations
        - Track optimization score improvements
        - Automated rebalancing recommendations
        - Performance vs benchmark analysis
        """)
        
        st.markdown("---")
        st.markdown("**⚠️ Disclaimer:** This tool is for educational and analysis purposes only. Past performance does not guarantee future results. Always consult with financial advisors before making investment decisions.")

if __name__ == "__main__":
    main()