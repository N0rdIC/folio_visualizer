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
    page_title="Portfolio Simulator & Analyzer", 
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

    def create_portfolio_backup(self, holdings: List[Dict], metrics: Dict = None, name: str = None) -> Dict:
        """Create a comprehensive portfolio backup"""
        backup = {
            'portfolio_name': name or st.session_state.portfolio_name,
            'portfolio_notes': st.session_state.portfolio_notes,
            'holdings': holdings,
            'created_date': datetime.now().isoformat(),
            'total_holdings': len(holdings),
            'app_version': '1.0.0'
        }
        
        if metrics:
            backup['summary_metrics'] = {
                'total_value': metrics.get('total_value', 0),
                'annual_dividends': metrics.get('annual_dividends', 0),
                'portfolio_dividend_yield': metrics.get('portfolio_dividend_yield', 0),
                'portfolio_beta': metrics.get('portfolio_beta', 1),
                'weighted_pe': metrics.get('weighted_pe', 0)
            }
        
        return backup

    def export_portfolio_json(self, holdings: List[Dict], metrics: Dict = None, name: str = None) -> str:
        """Export portfolio as JSON string"""
        backup = self.create_portfolio_backup(holdings, metrics, name)
        return json.dumps(backup, indent=2)

    def export_portfolio_csv(self, portfolio_df: pd.DataFrame) -> str:
        """Export portfolio as CSV string"""
        # Create a comprehensive CSV with current market data
        export_df = portfolio_df.copy()
        
        # Add calculated fields
        export_df['Current_Value'] = export_df['shares'] * export_df['current_price']
        export_df['Annual_Dividends'] = export_df['shares'] * export_df['dividend_rate']
        export_df['Weight_Percent'] = (export_df['Current_Value'] / export_df['Current_Value'].sum()) * 100
        
        # Select and order columns for export
        columns_to_export = [
            'symbol', 'name', 'shares', 'current_price', 'Current_Value', 'Weight_Percent',
            'sector', 'industry', 'country', 'dividend_yield', 'dividend_rate', 
            'Annual_Dividends', 'payout_ratio', 'pe_ratio', 'beta', 'market_cap'
        ]
        
        # Only include columns that exist
        available_columns = [col for col in columns_to_export if col in export_df.columns]
        
        return export_df[available_columns].to_csv(index=False)

    def import_portfolio_from_json(self, json_data: str) -> Tuple[List[Dict], Dict]:
        """Import portfolio from JSON string"""
        try:
            data = json.loads(json_data)
            
            # Validate JSON structure
            if 'holdings' not in data:
                raise ValueError("Invalid portfolio file: missing 'holdings' data")
            
            holdings = data['holdings']
            metadata = {
                'name': data.get('portfolio_name', 'Imported Portfolio'),
                'notes': data.get('portfolio_notes', ''),
                'created_date': data.get('created_date', ''),
                'total_holdings': data.get('total_holdings', len(holdings)),
                'summary_metrics': data.get('summary_metrics', {})
            }
            
            # Validate holdings structure
            for holding in holdings:
                if 'symbol' not in holding or 'shares' not in holding:
                    raise ValueError("Invalid holding format: missing 'symbol' or 'shares'")
            
            return holdings, metadata
            
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format")
        except Exception as e:
            raise ValueError(f"Error importing portfolio: {str(e)}")

    def import_portfolio_from_csv(self, csv_data: str) -> List[Dict]:
        """Import portfolio from CSV string"""
        try:
            df = pd.read_csv(io.StringIO(csv_data))
            
            # Check required columns
            if 'symbol' not in df.columns:
                raise ValueError("CSV must contain 'symbol' column")
            
            if 'shares' not in df.columns:
                raise ValueError("CSV must contain 'shares' column")
            
            # Convert to holdings format
            holdings = []
            for _, row in df.iterrows():
                holdings.append({
                    'symbol': str(row['symbol']).upper().strip(),
                    'shares': float(row['shares'])
                })
            
            return holdings
            
        except Exception as e:
            raise ValueError(f"Error importing CSV: {str(e)}")

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
        # In a real implementation, you could show a form for manual entry
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

    def fetch_stock_info(self, symbol: str, retry_count: int = 3, use_fallback: bool = True) -> Dict:
        """Fetch comprehensive stock information with rate limiting and fallback options"""
        
        # Add random delay to avoid hitting rate limits
        delay = random.uniform(0.5, 1.5)  # Random delay between 0.5-1.5 seconds
        time.sleep(delay)
        
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
                    # Exponential backoff for rate limiting
                    wait_time = (2 ** attempt) + random.uniform(1, 3)
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

    def clear_session_data(self):
        """Clear all session data"""
        st.session_state.portfolio_holdings = []
        st.session_state.last_analysis_date = None
        st.session_state.portfolio_name = "My Portfolio"
        st.session_state.portfolio_notes = ""

    def fetch_historical_data(self, symbol: str, period: str = "5y") -> pd.DataFrame:
        """Fetch historical price data"""
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period=period)
            
            # Get dividend data
            dividends = stock.dividends
            
            # Merge dividends with price data
            hist['Dividends'] = 0
            for date, div in dividends.items():
                if date in hist.index:
                    hist.loc[date, 'Dividends'] = div
            
            return hist
        except Exception as e:
            st.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return pd.DataFrame()

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

    def simulate_historical_performance(self, portfolio_holdings: List[Dict], years: int = 5, batch_size: int = 10, delay: int = 5) -> pd.DataFrame:
        """Simulate historical portfolio performance with total return (price + dividends)"""
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years*365)
        
        # Fetch historical data for all holdings
        all_data = {}
        total_initial_value = 0
        successful_fetches = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        batch_info = st.empty()
        
        # Process in batches
        for batch_start in range(0, len(portfolio_holdings), batch_size):
            batch_end = min(batch_start + batch_size, len(portfolio_holdings))
            current_batch = portfolio_holdings[batch_start:batch_end]
            batch_num = (batch_start // batch_size) + 1
            total_batches = (len(portfolio_holdings) + batch_size - 1) // batch_size
            
            batch_info.info(f"📈 Fetching historical data - Batch {batch_num}/{total_batches}")
            
            for i, holding in enumerate(current_batch):
                symbol = holding['symbol']
                shares = holding['shares']
                overall_progress = (batch_start + i + 1) / len(portfolio_holdings)
                
                status_text.text(f'Fetching {years}-year historical data for {symbol}...')
                progress_bar.progress(overall_progress)
                
                # Skip if no shares or invalid data
                if shares <= 0:
                    continue
                
                try:
                    # Add delay to avoid rate limiting
                    time.sleep(random.uniform(0.5, 1.0))
                    
                    stock = yf.Ticker(symbol)
                    
                    # Try to get historical data with multiple attempts
                    hist = None
                    for attempt in range(3):
                        try:
                            hist = stock.history(start=start_date, end=end_date, auto_adjust=True, back_adjust=True)
                            if not hist.empty and len(hist) > 10:  # Need at least 10 data points
                                break
                            time.sleep(1)  # Wait between attempts
                        except Exception as e:
                            if attempt == 2:  # Last attempt
                                st.warning(f"Failed to fetch historical data for {symbol} after 3 attempts: {str(e)}")
                            time.sleep(2)
                    
                    if hist is None or hist.empty:
                        st.warning(f"No historical data available for {symbol}")
                        continue
                    
                    # Ensure we have enough data
                    if len(hist) < 10:
                        st.warning(f"Insufficient historical data for {symbol} ({len(hist)} data points)")
                        continue
                    
                    # Calculate position value over time
                    hist['Position_Value'] = hist['Close'] * shares
                    
                    # Get dividend data and add to historical data
                    try:
                        dividends = stock.dividends
                        hist['Dividend_Income'] = 0
                        
                        if len(dividends) > 0:
                            # Add dividends received on each date
                            for date, div in dividends.items():
                                # Handle timezone issues
                                if hasattr(date, 'tz') and date.tz is not None:
                                    date = date.tz_localize(None)
                                
                                if date in hist.index:
                                    hist.loc[date, 'Dividend_Income'] = div * shares
                    except Exception as e:
                        st.warning(f"Could not fetch dividend data for {symbol}: {str(e)}")
                        hist['Dividend_Income'] = 0
                    
                    # Calculate cumulative dividends (total dividends received up to each date)
                    hist['Cumulative_Dividends'] = hist['Dividend_Income'].cumsum()
                    
                    # Calculate total return percentage (capital gains + dividends)
                    initial_price = hist['Close'].iloc[0]
                    if initial_price <= 0:
                        st.warning(f"Invalid initial price for {symbol}: {initial_price}")
                        continue
                        
                    initial_investment = initial_price * shares
                    
                    # Total value = current position value + all dividends received
                    hist['Total_Value'] = hist['Position_Value'] + hist['Cumulative_Dividends']
                    
                    # Total return percentage = (total value / initial investment - 1) * 100
                    hist['Total_Return_Pct'] = ((hist['Total_Value'] / initial_investment) - 1) * 100
                    
                    # Price-only return (for comparison)
                    hist['Price_Return_Pct'] = ((hist['Close'] / initial_price) - 1) * 100
                    
                    all_data[symbol] = hist
                    total_initial_value += initial_investment
                    successful_fetches += 1
                    
                except Exception as e:
                    st.warning(f"Error processing historical data for {symbol}: {str(e)}")
                    continue
            
            # Wait between batches (except for the last batch)
            if batch_end < len(portfolio_holdings):
                status_text.text(f'Waiting {delay}s before next batch...')
                time.sleep(delay)
        
        progress_bar.empty()
        status_text.empty()
        batch_info.empty()
        
        # Check if we have enough data
        if len(all_data) == 0:
            st.error("❌ No historical data could be fetched for any stocks in your portfolio.")
            st.info("💡 This could be due to:")
            st.write("• Rate limiting from Yahoo Finance")
            st.write("• Invalid stock symbols")
            st.write("• Network connectivity issues")
            st.write("• Try using backup APIs or wait before retrying")
            return pd.DataFrame()
        
        if successful_fetches < len(portfolio_holdings) * 0.5:
            st.warning(f"⚠️ Only {successful_fetches}/{len(portfolio_holdings)} stocks had historical data fetched. Results may be incomplete.")
        
        # Combine all positions into portfolio performance
        portfolio_performance = pd.DataFrame()
        
        # Get common date range (intersection of all stock data)
        common_dates = None
        for symbol, data in all_data.items():
            if common_dates is None:
                common_dates = data.index
            else:
                common_dates = common_dates.intersection(data.index)
        
        if len(common_dates) == 0:
            st.error("❌ No common date range found across stocks.")
            return pd.DataFrame()
        
        if len(common_dates) < 50:
            st.warning(f"⚠️ Limited common date range: {len(common_dates)} data points")
        
        # Calculate portfolio metrics for each date
        for date in common_dates:
            portfolio_value = 0
            total_dividends = 0
            
            for symbol, data in all_data.items():
                if date in data.index:
                    portfolio_value += data.loc[date, 'Position_Value']
                    total_dividends += data.loc[date, 'Cumulative_Dividends']
            
            portfolio_performance.loc[date, 'Portfolio_Value'] = portfolio_value
            portfolio_performance.loc[date, 'Total_Dividends'] = total_dividends
            portfolio_performance.loc[date, 'Total_Value'] = portfolio_value + total_dividends
            
            # Total return including dividends
            if total_initial_value > 0:
                portfolio_performance.loc[date, 'Total_Return'] = ((portfolio_value + total_dividends) / total_initial_value - 1) * 100
                portfolio_performance.loc[date, 'Price_Only_Return'] = (portfolio_value / total_initial_value - 1) * 100
            else:
                portfolio_performance.loc[date, 'Total_Return'] = 0
                portfolio_performance.loc[date, 'Price_Only_Return'] = 0
        
        # Success message
        st.success(f"✅ Historical analysis complete! {successful_fetches} stocks analyzed over {len(common_dates)} trading days.")
        
        return portfolio_performance.sort_index()

    def generate_historical_dividend_calendar(self, portfolio_holdings: List[Dict], batch_size: int = 10, delay: int = 2) -> pd.DataFrame:
        """Generate historical dividend calendar showing actual payments from last year"""
        dividend_history = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Get date range for last year
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # Process in batches for dividend calendar
        for batch_start in range(0, len(portfolio_holdings), batch_size):
            batch_end = min(batch_start + batch_size, len(portfolio_holdings))
            current_batch = portfolio_holdings[batch_start:batch_end]
            
            for i, holding in enumerate(current_batch):
                symbol = holding['symbol']
                shares = holding['shares']
                overall_progress = (batch_start + i + 1) / len(portfolio_holdings)
                
                status_text.text(f'Fetching dividend history for {symbol}...')
                progress_bar.progress(overall_progress)
                
                try:
                    # Add delay to avoid rate limiting
                    time.sleep(random.uniform(0.3, 0.8))
                    
                    stock = yf.Ticker(symbol)
                    
                    # Get dividend history for the last year
                    dividends = stock.dividends
                    
                    # Filter dividends from last year and handle timezone issues
                    if len(dividends) > 0:
                        # Convert timezone-aware index to timezone-naive for comparison
                        dividend_dates = dividends.index.tz_localize(None) if dividends.index.tz is not None else dividends.index
                        
                        # Filter dividends from the last year
                        recent_mask = (dividend_dates >= start_date) & (dividend_dates <= end_date)
                        recent_dividends = dividends[recent_mask]
                        recent_dividend_dates = dividend_dates[recent_mask]
                        
                        # Add each dividend payment to history
                        for date, dividend_amount in zip(recent_dividend_dates, recent_dividends):
                            dividend_history.append({
                                'Date': date.date(),
                                'Symbol': symbol,
                                'Company': f"{symbol}",  # We'll get company name from stock info if available
                                'Dividend_Per_Share': float(dividend_amount),
                                'Shares_Owned': shares,
                                'Total_Dividend_Received': float(dividend_amount) * shares,
                                'Month': date.strftime('%Y-%m'),
                                'Quarter': f"Q{((date.month-1)//3)+1} {date.year}",
                                'Day_of_Year': date.timetuple().tm_yday,
                                'Weekday': date.strftime('%A')
                            })
                
                except Exception as e:
                    st.warning(f"Could not fetch dividend history for {symbol}: {str(e)}")
            
            # Wait between batches (except for the last batch)
            if batch_end < len(portfolio_holdings):
                time.sleep(delay)
        
        progress_bar.empty()
        status_text.empty()
        
        if dividend_history:
            df = pd.DataFrame(dividend_history)
            
            # Sort by date
            df = df.sort_values('Date')
            
            # Create daily aggregation for plotting
            daily_dividends = df.groupby('Date').agg({
                'Total_Dividend_Received': 'sum',
                'Symbol': lambda x: ', '.join(x.unique()),
                'Dividend_Per_Share': 'sum'  # This isn't quite right but gives a sense of magnitude
            }).reset_index()
            daily_dividends.rename(columns={'Symbol': 'Companies'}, inplace=True)
            
            # Create monthly aggregation
            monthly_dividends = df.groupby('Month').agg({
                'Total_Dividend_Received': 'sum',
                'Symbol': lambda x: len(x.unique()),
                'Dividend_Per_Share': 'count'
            }).reset_index()
            monthly_dividends.rename(columns={
                'Symbol': 'Companies_Count',
                'Dividend_Per_Share': 'Payment_Count'
            }, inplace=True)
            
            return df, daily_dividends, monthly_dividends
        else:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def analyze_portfolio_with_ai(self, portfolio_data: pd.DataFrame, metrics: Dict) -> str:
        """Generate AI-powered portfolio analysis"""
        
        if not self.openai_api_key:
            return self._generate_rule_based_analysis(portfolio_data, metrics)
        
        try:
            # Prepare portfolio summary for AI
            portfolio_summary = {
                'total_value': metrics['total_value'],
                'dividend_yield': metrics['portfolio_dividend_yield'],
                'beta': metrics['portfolio_beta'],
                'pe_ratio': metrics['weighted_pe'],
                'annual_dividends': metrics['annual_dividends'],
                'holdings_count': len(portfolio_data),
                'top_holdings': metrics['top_holdings'][['symbol', 'weight', 'sector']].head(5).to_dict('records'),
                'sector_allocation': metrics['sector_allocation']['weight'].to_dict()
            }
            
            # Create prompt for OpenAI
            prompt = f"""
            As a professional financial advisor, analyze this investment portfolio and provide insights:
            
            Portfolio Summary:
            - Total Value: ${metrics['total_value']:,.2f}
            - Portfolio Dividend Yield: {metrics['portfolio_dividend_yield']:.2f}%
            - Annual Dividend Income: ${metrics['annual_dividends']:,.2f}
            - Portfolio Beta: {metrics['portfolio_beta']:.2f}
            - Weighted P/E Ratio: {metrics['weighted_pe']:.2f}
            - Number of Holdings: {len(portfolio_data)}
            
            Top Holdings: {portfolio_summary['top_holdings']}
            
            Sector Allocation: {portfolio_summary['sector_allocation']}
            
            Please provide:
            1. Overall portfolio assessment (strengths and weaknesses)
            2. Diversification analysis
            3. Risk assessment
            4. Dividend sustainability analysis
            5. Specific recommendations for improvement
            6. Potential red flags or concerns
            
            Keep the analysis concise but comprehensive, suitable for an investor review.
            """
            
            # Note: In a real implementation, you would call OpenAI API here
            # For this demo, we'll use the rule-based analysis
            return self._generate_rule_based_analysis(portfolio_data, metrics)
            
        except Exception as e:
            st.warning(f"AI analysis unavailable: {str(e)}")
            return self._generate_rule_based_analysis(portfolio_data, metrics)

    def _generate_rule_based_analysis(self, portfolio_data: pd.DataFrame, metrics: Dict) -> str:
        """Generate rule-based portfolio analysis"""
        
        analysis = []
        
        # Overall Assessment
        analysis.append("## 📊 Overall Portfolio Assessment")
        
        total_value = metrics['total_value']
        dividend_yield = metrics['portfolio_dividend_yield']
        beta = metrics['portfolio_beta']
        
        if dividend_yield > 4:
            analysis.append(f"✅ **Strong Dividend Income**: Your portfolio yields {dividend_yield:.2f}%, providing solid passive income.")
        elif dividend_yield > 2:
            analysis.append(f"⚡ **Moderate Dividend Income**: Your portfolio yields {dividend_yield:.2f}%, which is reasonable for income generation.")
        else:
            analysis.append(f"⚠️ **Low Dividend Yield**: At {dividend_yield:.2f}%, your portfolio may not provide sufficient income for dividend investors.")
        
        # Risk Assessment
        analysis.append("\n## ⚠️ Risk Assessment")
        
        if beta < 0.8:
            analysis.append(f"🛡️ **Low Risk Profile**: Portfolio beta of {beta:.2f} suggests lower volatility than the market.")
        elif beta > 1.2:
            analysis.append(f"📈 **High Risk Profile**: Portfolio beta of {beta:.2f} indicates higher volatility than the market.")
        else:
            analysis.append(f"⚖️ **Moderate Risk Profile**: Portfolio beta of {beta:.2f} is close to market volatility.")
        
        # Diversification Analysis
        analysis.append("\n## 🎯 Diversification Analysis")
        
        sector_allocation = metrics['sector_allocation']
        num_sectors = len(sector_allocation)
        max_sector_weight = sector_allocation['weight'].max()
        
        if num_sectors >= 6:
            analysis.append(f"✅ **Good Sector Diversification**: Portfolio spans {num_sectors} sectors.")
        elif num_sectors >= 3:
            analysis.append(f"⚡ **Moderate Diversification**: Portfolio covers {num_sectors} sectors - consider adding more.")
        else:
            analysis.append(f"⚠️ **Poor Diversification**: Only {num_sectors} sectors represented - high concentration risk.")
        
        if max_sector_weight > 0.4:
            analysis.append(f"⚠️ **Sector Concentration Risk**: Largest sector represents {max_sector_weight*100:.1f}% of portfolio.")
        
        # Top Holdings Analysis
        analysis.append("\n## 🏆 Top Holdings Analysis")
        
        top_holdings = metrics['top_holdings']
        largest_position = top_holdings['weight'].iloc[0]
        
        if largest_position > 0.15:
            analysis.append(f"⚠️ **Position Concentration**: Largest holding ({top_holdings['symbol'].iloc[0]}) represents {largest_position*100:.1f}% of portfolio.")
        
        top_5_weight = top_holdings['weight'].head(5).sum()
        analysis.append(f"📈 **Top 5 Holdings**: Represent {top_5_weight*100:.1f}% of total portfolio value.")
        
        # Dividend Sustainability
        analysis.append("\n## 💰 Dividend Sustainability")
        
        avg_payout_ratio = portfolio_data['payout_ratio'].mean()
        if avg_payout_ratio > 80:
            analysis.append(f"⚠️ **High Payout Ratios**: Average payout ratio of {avg_payout_ratio:.1f}% may indicate dividend sustainability risks.")
        elif avg_payout_ratio > 60:
            analysis.append(f"⚡ **Moderate Payout Ratios**: Average payout ratio of {avg_payout_ratio:.1f}% is reasonable but worth monitoring.")
        else:
            analysis.append(f"✅ **Conservative Payout Ratios**: Average payout ratio of {avg_payout_ratio:.1f}% suggests sustainable dividends.")
        
        # Recommendations
        analysis.append("\n## 🎯 Recommendations")
        
        recommendations = []
        
        if num_sectors < 5:
            recommendations.append("• **Increase sector diversification** by adding positions in underrepresented sectors")
        
        if max_sector_weight > 0.4:
            recommendations.append("• **Reduce sector concentration** by rebalancing overweight positions")
        
        if largest_position > 0.15:
            recommendations.append("• **Consider reducing position size** in largest holdings to manage concentration risk")
        
        if dividend_yield < 3:
            recommendations.append("• **Add higher-yielding dividend stocks** to increase income generation")
        
        if avg_payout_ratio > 70:
            recommendations.append("• **Review companies with high payout ratios** for dividend sustainability")
        
        if beta > 1.3:
            recommendations.append("• **Consider adding defensive stocks** to reduce overall portfolio volatility")
        
        if len(recommendations) == 0:
            recommendations.append("• **Well-balanced portfolio** - continue monitoring and periodic rebalancing")
        
        analysis.extend(recommendations)
        
        return "\n".join(analysis)

def main():
    st.title("💼 Portfolio Simulator & Analyzer")
    st.markdown("**Comprehensive portfolio analysis with AI-powered insights**")
    
    analyzer = PortfolioAnalyzer()
    
    # Sidebar for portfolio management and input
    st.sidebar.header("💾 Portfolio Management")
    
    # Portfolio session info
    if st.session_state.last_analysis_date:
        last_date = datetime.fromisoformat(st.session_state.last_analysis_date)
        st.sidebar.info(f"📅 Last analysis: {last_date.strftime('%Y-%m-%d %H:%M')}")
        
        if st.session_state.portfolio_holdings:
            st.sidebar.success(f"💼 Saved portfolio: {len(st.session_state.portfolio_holdings)} holdings")
    
    # Portfolio management options
    portfolio_action = st.sidebar.radio(
        "Portfolio Action:",
        ["📝 Create/Edit Portfolio", "📂 Load Saved Portfolio", "📁 Import Portfolio File", "🗑️ Clear Session"]
    )
    
    portfolio_holdings = []
    
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
                st.sidebar.success("Loaded saved portfolio!")
        else:
            st.sidebar.info("No saved portfolio found. Create one first!")
    
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
                    imported_holdings = analyzer.import_portfolio_from_csv(file_content)
                    portfolio_holdings = imported_holdings
                    st.sidebar.success(f"✅ Imported {len(imported_holdings)} holdings from CSV")
                    
                else:  # JSON Format
                    imported_holdings, metadata = analyzer.import_portfolio_from_json(file_content)
                    portfolio_holdings = imported_holdings
                    
                    # Update session with imported metadata
                    st.session_state.portfolio_name = metadata['name']
                    st.session_state.portfolio_notes = metadata['notes']
                    
                    st.sidebar.success(f"✅ Imported portfolio: {metadata['name']}")
                    if metadata['created_date']:
                        st.sidebar.info(f"Created: {metadata['created_date'][:10]}")
                
            except Exception as e:
                st.sidebar.error(f"❌ Import failed: {str(e)}")
    
    else:  # Create/Edit Portfolio
        st.sidebar.header("📝 Portfolio Input")
        
        # Portfolio metadata
        portfolio_name = st.sidebar.text_input(
            "Portfolio Name", 
            value=st.session_state.portfolio_name,
            placeholder="e.g., Dividend Growth Portfolio"
        )
        
        portfolio_notes = st.sidebar.text_area(
            "Portfolio Notes", 
            value=st.session_state.portfolio_notes,
            placeholder="Investment strategy, goals, notes..."
        )
        
        # Update session state
        st.session_state.portfolio_name = portfolio_name
        st.session_state.portfolio_notes = portfolio_notes
        
        # Portfolio input methods
        input_method = st.sidebar.radio(
            "Choose input method:",
            ["Manual Entry", "Upload CSV", "Sample Portfolio"]
        )
        
        if input_method == "Manual Entry":
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
            
            # Display and manage current holdings
            if st.session_state.holdings:
                st.sidebar.subheader("Current Holdings")
                for i, holding in enumerate(st.session_state.holdings):
                    col1, col2 = st.sidebar.columns([3, 1])
                    col1.write(f"{holding['symbol']}: {holding['shares']} shares")
                    if col2.button("❌", key=f"remove_{i}"):
                        st.session_state.holdings.pop(i)
                        st.rerun()
                
                # Save to session button
                if st.sidebar.button("💾 Save Portfolio", type="primary"):
                    analyzer.save_portfolio_to_session(
                        st.session_state.holdings, 
                        portfolio_name, 
                        portfolio_notes
                    )
                    st.sidebar.success("Portfolio saved to session!")
            
            portfolio_holdings = st.session_state.holdings
        
        elif input_method == "Upload CSV":
            st.sidebar.subheader("Upload Portfolio CSV")
            uploaded_file = st.sidebar.file_uploader(
                "Choose CSV file", 
                type="csv",
                help="CSV should have columns: symbol, shares"
            )
            
            if uploaded_file:
                try:
                    csv_content = uploaded_file.read().decode('utf-8')
                    imported_holdings = analyzer.import_portfolio_from_csv(csv_content)
                    portfolio_holdings = imported_holdings
                    st.sidebar.success(f"Loaded {len(portfolio_holdings)} holdings")
                    
                    # Auto-save to session
                    analyzer.save_portfolio_to_session(portfolio_holdings, portfolio_name, portfolio_notes)
                    
                except Exception as e:
                    st.sidebar.error(f"Error reading CSV: {str(e)}")
        
        else:  # Sample Portfolio
            st.sidebar.subheader("Sample Dividend Portfolio")
            portfolio_holdings = [
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
    
    # Analysis configuration
    st.sidebar.header("⚙️ Analysis Settings")
    analysis_period = st.sidebar.selectbox(
        "Historical Analysis Period",
        ["1y", "2y", "3y", "5y", "10y"],
        index=3,
        help="Longer periods provide better trend analysis but take more time to load"
    )
    
    # Rate limiting options
    st.sidebar.subheader("⏱️ Rate Limiting")
    batch_size = st.sidebar.select_slider(
        "Batch Size",
        options=[5, 10, 15, 20, 25],
        value=10,
        help="Process stocks in smaller batches to avoid rate limits"
    )
    
    delay_between_batches = st.sidebar.select_slider(
        "Delay Between Batches (seconds)",
        options=[2, 5, 10, 15, 20],
        value=5,
        help="Wait time between batches to respect API limits"
    )
    
    include_ai_analysis = st.sidebar.checkbox(
        "Include AI Analysis", 
        value=True,
        help="Requires OpenAI API key in secrets"
    )
    
    # Cache options
    st.sidebar.subheader("💾 Cache Options")
    use_cache = st.sidebar.checkbox(
        "Use Session Cache",
        value=True,
        help="Cache stock data to avoid repeated API calls"
    )
    
    if st.sidebar.button("🗑️ Clear Cache"):
        if 'stock_data_cache' in st.session_state:
            st.session_state.stock_data_cache = {}
            st.sidebar.success("Cache cleared!")
    
    # Data source options
    st.sidebar.subheader("🔄 Data Sources")
    use_fallback_apis = st.sidebar.checkbox(
        "Enable Fallback APIs",
        value=True,
        help="Use Alpha Vantage/FMP when Yahoo Finance fails"
    )
    
    # API status indicators
    yf_status = "🟢 Available"
    av_status = "🔴 No API Key" if not st.secrets.get("ALPHA_VANTAGE_API_KEY") else "🟢 Available"
    fmp_status = "🔴 No API Key" if not st.secrets.get("FMP_API_KEY") else "🟢 Available"
    
    st.sidebar.write(f"**Yahoo Finance:** {yf_status}")
    st.sidebar.write(f"**Alpha Vantage:** {av_status}")
    st.sidebar.write(f"**FMP:** {fmp_status}")
    
    # Ban recovery helper
    st.sidebar.subheader("🚨 Rate Limit Recovery")
    if st.sidebar.button("🧪 Test Connection"):
        test_symbol = "AAPL"
        try:
            test_data = analyzer.fetch_stock_info(test_symbol, retry_count=1, use_fallback=False)
            if test_data['current_price'] > 0:
                st.sidebar.success("✅ Yahoo Finance working!")
            else:
                st.sidebar.error("❌ Still rate limited")
        except:
            st.sidebar.error("❌ Still rate limited")
    
    st.sidebar.info("💡 If rate limited, wait 2-6 hours or enable fallback APIs")
    
    # Main analysis
    if portfolio_holdings and st.sidebar.button("🚀 Analyze Portfolio", type="primary"):
        
        # Initialize cache if using it
        if use_cache and 'stock_data_cache' not in st.session_state:
            st.session_state.stock_data_cache = {}
        
        st.info(f"Analyzing portfolio with {len(portfolio_holdings)} holdings using batch processing...")
        
        # Show rate limiting info
        estimated_time = (len(portfolio_holdings) / batch_size) * delay_between_batches
        st.info(f"⏱️ Processing in batches of {batch_size} with {delay_between_batches}s delays. Estimated time: {estimated_time:.0f}s")
        
        # Fetch current data for all holdings with batch processing
        portfolio_data = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        batch_info = st.empty()
        
        # Process in batches to avoid rate limiting
        for batch_start in range(0, len(portfolio_holdings), batch_size):
            batch_end = min(batch_start + batch_size, len(portfolio_holdings))
            current_batch = portfolio_holdings[batch_start:batch_end]
            batch_num = (batch_start // batch_size) + 1
            total_batches = (len(portfolio_holdings) + batch_size - 1) // batch_size
            
            batch_info.info(f"📦 Processing batch {batch_num}/{total_batches} ({len(current_batch)} stocks)")
            
            for i, holding in enumerate(current_batch):
                symbol = holding['symbol']
                overall_progress = (batch_start + i + 1) / len(portfolio_holdings)
                
                status_text.text(f'Fetching data for {symbol}... ({batch_start + i + 1}/{len(portfolio_holdings)})')
                progress_bar.progress(overall_progress)
                
                # Check cache first
                if use_cache and symbol in st.session_state.stock_data_cache:
                    # Check if cached data is recent (less than 1 hour old)
                    cached_data = st.session_state.stock_data_cache[symbol]
                    if 'cache_time' in cached_data:
                        cache_time = datetime.fromisoformat(cached_data['cache_time'])
                        if datetime.now() - cache_time < timedelta(hours=1):
                            stock_info = cached_data.copy()
                            stock_info.pop('cache_time', None)  # Remove cache timestamp
                            stock_info['shares'] = holding['shares']
                            portfolio_data.append(stock_info)
                            status_text.text(f'Using cached data for {symbol}... ({batch_start + i + 1}/{len(portfolio_holdings)})')
                            continue
                
                # Fetch new data
                stock_info = analyzer.fetch_stock_info(symbol, use_fallback=use_fallback_apis)
                stock_info['shares'] = holding['shares']
                portfolio_data.append(stock_info)
                
                # Cache the data if using cache
                if use_cache and stock_info['current_price'] > 0:  # Only cache successful fetches
                    cache_data = stock_info.copy()
                    cache_data['cache_time'] = datetime.now().isoformat()
                    st.session_state.stock_data_cache[symbol] = cache_data
            
            # Wait between batches (except for the last batch)
            if batch_end < len(portfolio_holdings):
                status_text.text(f'Waiting {delay_between_batches}s before next batch...')
                time.sleep(delay_between_batches)
        
        progress_bar.empty()
        status_text.empty()
        batch_info.empty()
        
        portfolio_df = pd.DataFrame(portfolio_data)
        
        # Calculate portfolio metrics
        metrics = analyzer.calculate_portfolio_metrics(portfolio_df)
        
        # Auto-save analyzed portfolio to session
        analyzer.save_portfolio_to_session(
            portfolio_holdings, 
            st.session_state.portfolio_name, 
            st.session_state.portfolio_notes
        )
        
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
        
        # Advanced Risk Metrics
        st.subheader("📈 Advanced Risk Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            alpha_color = "normal" if abs(metrics['portfolio_alpha']) < 0.02 else ("inverse" if metrics['portfolio_alpha'] > 0 else "off")
            st.metric(
                "Portfolio Alpha", 
                f"{metrics['portfolio_alpha']:.3f}",
                help="Excess return vs market (positive = outperforming)"
            )
        
        with col2:
            sharpe_color = "normal" if metrics['sharpe_ratio'] > 1.0 else "off"
            st.metric(
                "Sharpe Ratio", 
                f"{metrics['sharpe_ratio']:.2f}",
                help="Risk-adjusted return (>1.0 = good, >2.0 = excellent)"
            )
        
        with col3:
            st.metric(
                "Information Ratio", 
                f"{metrics['information_ratio']:.2f}",
                help="Alpha per unit of tracking error"
            )
        
        with col4:
            volatility_estimate = metrics['portfolio_beta'] * 16  # Rough volatility estimate
            st.metric(
                "Est. Volatility", 
                f"{volatility_estimate:.1f}%",
                help="Estimated annual volatility"
            )
        
        # Risk interpretation
        st.info(f"""
        **🎯 Risk Profile Summary:**
        - **Beta {metrics['portfolio_beta']:.2f}**: {'Lower' if metrics['portfolio_beta'] < 0.8 else 'Higher' if metrics['portfolio_beta'] > 1.2 else 'Similar'} risk vs market
        - **Alpha {metrics['portfolio_alpha']:.3f}**: {'Outperforming' if metrics['portfolio_alpha'] > 0.01 else 'Underperforming' if metrics['portfolio_alpha'] < -0.01 else 'Matching'} market expectations
        - **Sharpe {metrics['sharpe_ratio']:.2f}**: {'Excellent' if metrics['sharpe_ratio'] > 2.0 else 'Good' if metrics['sharpe_ratio'] > 1.0 else 'Poor'} risk-adjusted returns
        """)
        
        # Holdings table
        st.header("📋 Current Holdings")
        
        display_df = portfolio_df.copy()
        display_df['Current Value'] = display_df['shares'] * display_df['current_price']
        display_df['Weight'] = (display_df['Current Value'] / metrics['total_value']) * 100
        display_df['Annual Dividends'] = display_df['shares'] * display_df['dividend_rate']
        
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
        
        # Row 1: Sector allocation and top holdings
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Sector Allocation")
            sector_data = metrics['sector_allocation'].reset_index()
            sector_data['value'] = sector_data['shares']
            
            fig_sector = px.pie(
                sector_data, 
                values='weight', 
                names='sector',
                title="Portfolio Allocation by Sector"
            )
            st.plotly_chart(fig_sector, use_container_width=True)
        
        with col2:
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
        
        # Historical Performance Analysis
        st.header("📊 Historical Performance Analysis")
        st.info("💡 **Total Return** includes both price appreciation AND dividend reinvestment")
        
        with st.spinner("Calculating historical performance..."):
            years = int(analysis_period[0]) if analysis_period[0].isdigit() else int(analysis_period[:2])
            historical_performance = analyzer.simulate_historical_performance(
                portfolio_holdings, 
                years, 
                batch_size=batch_size, 
                delay=delay_between_batches
            )
        
        if not historical_performance.empty:
            # Performance chart with both total return and price-only return
            fig_performance = make_subplots(
                rows=3, cols=1,
                subplot_titles=(
                    f'Portfolio Value Over Time ({years} Years)', 
                    'Total Return vs Price-Only Return (%)',
                    'Cumulative Dividends Received'
                ),
                vertical_spacing=0.08
            )
            
            # Portfolio value
            fig_performance.add_trace(
                go.Scatter(
                    x=historical_performance.index,
                    y=historical_performance['Portfolio_Value'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            # Total return vs price-only return
            fig_performance.add_trace(
                go.Scatter(
                    x=historical_performance.index,
                    y=historical_performance['Total_Return'],
                    mode='lines',
                    name='Total Return (w/ Dividends)',
                    line=dict(color='green', width=2)
                ),
                row=2, col=1
            )
            
            if 'Price_Only_Return' in historical_performance.columns:
                fig_performance.add_trace(
                    go.Scatter(
                        x=historical_performance.index,
                        y=historical_performance['Price_Only_Return'],
                        mode='lines',
                        name='Price-Only Return',
                        line=dict(color='orange', width=2, dash='dash')
                    ),
                    row=2, col=1
                )
            
            # Cumulative dividends
            fig_performance.add_trace(
                go.Scatter(
                    x=historical_performance.index,
                    y=historical_performance['Total_Dividends'],
                    mode='lines',
                    name='Cumulative Dividends',
                    line=dict(color='purple', width=2),
                    fill='tonexty'
                ),
                row=3, col=1
            )
            
            fig_performance.update_layout(height=800, showlegend=True)
            fig_performance.update_xaxes(title_text="Date", row=3, col=1)
            fig_performance.update_yaxes(title_text="Value ($)", row=1, col=1)
            fig_performance.update_yaxes(title_text="Return (%)", row=2, col=1)
            fig_performance.update_yaxes(title_text="Dividends ($)", row=3, col=1)
            
            st.plotly_chart(fig_performance, use_container_width=True)
            
            # Performance summary with dividend impact
            if len(historical_performance) > 1:
                total_return = historical_performance['Total_Return'].iloc[-1]
                price_only_return = historical_performance['Price_Only_Return'].iloc[-1] if 'Price_Only_Return' in historical_performance.columns else 0
                dividend_contribution = total_return - price_only_return
                annual_total_return = (1 + total_return/100) ** (1/years) - 1
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        "Total Return", 
                        f"{total_return:.1f}%",
                        help="Includes price appreciation + dividends"
                    )
                with col2:
                    st.metric(
                        "Annualized Return", 
                        f"{annual_total_return*100:.1f}%",
                        help="Compound annual growth rate"
                    )
                with col3:
                    final_dividends = historical_performance['Total_Dividends'].iloc[-1]
                    st.metric(
                        "Total Dividends", 
                        f"${final_dividends:,.0f}",
                        help="Cumulative dividends received"
                    )
                with col4:
                    st.metric(
                        "Dividend Contribution", 
                        f"{dividend_contribution:.1f}%",
                        help="Return boost from dividends"
                    )
                
                # Explanation of total return
                st.info(f"""
                📈 **{years}-Year Performance Summary:**
                - **Total Return**: {total_return:.1f}% (includes dividends reinvested)
                - **Price-Only Return**: {price_only_return:.1f}% (capital appreciation only)
                - **Dividend Boost**: {dividend_contribution:.1f}% additional return from dividends
                - **Annual Compound Rate**: {annual_total_return*100:.1f}% per year
                """)
        else:
            st.warning("Unable to fetch sufficient historical data for performance analysis.")
        
        # Dividend Calendar - Historical Analysis
        st.header("📅 Historical Dividend Calendar (Last 12 Months)")
        st.info("💰 **Actual dividend payments received** - showing real dividend income history")
        
        dividend_history, daily_dividends, monthly_dividends = analyzer.generate_historical_dividend_calendar(
            portfolio_holdings, 
            batch_size=batch_size, 
            delay=2  # Shorter delay for dividend calendar
        )
        
        if not dividend_history.empty:
            # Summary metrics
            total_dividends_received = dividend_history['Total_Dividend_Received'].sum()
            avg_monthly_dividends = monthly_dividends['Total_Dividend_Received'].mean() if not monthly_dividends.empty else 0
            dividend_payments_count = len(dividend_history)
            unique_companies = dividend_history['Symbol'].nunique()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Dividends Received", f"${total_dividends_received:,.2f}")
            with col2:
                st.metric("Average Monthly", f"${avg_monthly_dividends:,.2f}")
            with col3:
                st.metric("Total Payments", dividend_payments_count)
            with col4:
                st.metric("Dividend-Paying Stocks", unique_companies)
            
            # Main visualizations
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📊 Daily Dividend Payments")
                
                if not daily_dividends.empty:
                    # Create detailed daily bar chart
                    fig_daily = go.Figure()
                    
                    # Add bars for daily dividends
                    fig_daily.add_trace(go.Bar(
                        x=daily_dividends['Date'],
                        y=daily_dividends['Total_Dividend_Received'],
                        text=daily_dividends['Companies'],
                        textposition='outside',
                        hovertemplate='<br>'.join([
                            'Date: %{x}',
                            'Dividend: $%{y:.2f}',
                            'Companies: %{text}',
                            '<extra></extra>'
                        ]),
                        marker_color='green',
                        name='Daily Dividends'
                    ))
                    
                    fig_daily.update_layout(
                        title="Daily Dividend Payments - Last 12 Months",
                        xaxis_title="Date",
                        yaxis_title="Dividend Amount ($)",
                        hovermode='x',
                        height=400,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_daily, use_container_width=True)
                
                # Monthly aggregation chart with detailed hover
                if not monthly_dividends.empty:
                    st.subheader("📈 Monthly Dividend Summary")
                    
                    # Create detailed monthly data with stock information
                    monthly_detail = dividend_history.groupby('Month').agg({
                        'Total_Dividend_Received': 'sum',
                        'Symbol': lambda x: ', '.join(sorted(x.unique())),
                        'Dividend_Per_Share': 'count'
                    }).reset_index()
                    monthly_detail.rename(columns={
                        'Symbol': 'Paying_Stocks',
                        'Dividend_Per_Share': 'Payment_Count'
                    }, inplace=True)
                    
                    fig_monthly = go.Figure()
                    
                    fig_monthly.add_trace(go.Bar(
                        x=monthly_detail['Month'],
                        y=monthly_detail['Total_Dividend_Received'],
                        text=monthly_detail['Total_Dividend_Received'].apply(lambda x: f'${x:.0f}'),
                        textposition='outside',
                        hovertemplate='<br>'.join([
                            '<b>%{x}</b>',
                            'Total Dividends: $%{y:.2f}',
                            'Paying Stocks: %{customdata[0]}',
                            'Number of Payments: %{customdata[1]}',
                            '<extra></extra>'
                        ]),
                        customdata=monthly_detail[['Paying_Stocks', 'Payment_Count']],
                        marker_color='darkgreen',
                        name='Monthly Dividends'
                    ))
                    
                    fig_monthly.update_layout(
                        title="Monthly Dividend Income - Hover for Stock Details",
                        xaxis_title="Month",
                        yaxis_title="Dividend Amount ($)",
                        height=300,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_monthly, use_container_width=True)
            
            with col2:
                st.subheader("🏢 Dividend by Company")
                
                # Company dividend summary
                company_summary = dividend_history.groupby('Symbol').agg({
                    'Total_Dividend_Received': 'sum',
                    'Dividend_Per_Share': 'count',  # Number of payments
                    'Date': 'max'  # Last payment date
                }).reset_index()
                company_summary.rename(columns={
                    'Dividend_Per_Share': 'Payment_Count',
                    'Date': 'Last_Payment'
                }, inplace=True)
                company_summary = company_summary.sort_values('Total_Dividend_Received', ascending=False)
                
                # Company pie chart
                fig_pie = px.pie(
                    company_summary.head(10),  # Top 10 to avoid clutter
                    values='Total_Dividend_Received',
                    names='Symbol',
                    title="Dividend Distribution by Company"
                )
                fig_pie.update_layout(height=300)
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # Top dividend payers table
                st.subheader("🏆 Top Dividend Payers")
                top_payers = company_summary.head(10).copy()
                top_payers['Total_Dividend_Received'] = top_payers['Total_Dividend_Received'].apply(lambda x: f"${x:.2f}")
                top_payers['Last_Payment'] = top_payers['Last_Payment'].apply(lambda x: x.strftime('%Y-%m-%d'))
                
                st.dataframe(
                    top_payers[['Symbol', 'Total_Dividend_Received', 'Payment_Count', 'Last_Payment']],
                    column_config={
                        "Symbol": "Stock",
                        "Total_Dividend_Received": "Total Received",
                        "Payment_Count": "# Payments",
                        "Last_Payment": "Last Payment"
                    },
                    use_container_width=True,
                    hide_index=True
                )
            
            # Detailed dividend history table
            st.subheader("📋 Complete Dividend Payment History")
            
            # Format the dividend history for display
            display_history = dividend_history.copy()
            display_history['Date'] = display_history['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
            display_history['Dividend_Per_Share'] = display_history['Dividend_Per_Share'].apply(lambda x: f"${x:.4f}")
            display_history['Total_Dividend_Received'] = display_history['Total_Dividend_Received'].apply(lambda x: f"${x:.2f}")
            
            # Sort by date (most recent first)
            display_history = display_history.sort_values('Date', ascending=False)
            
            # Show recent payments with expandable view
            with st.expander("📜 View All Dividend Payments", expanded=False):
                st.dataframe(
                    display_history[['Date', 'Symbol', 'Dividend_Per_Share', 'Shares_Owned', 'Total_Dividend_Received', 'Quarter']],
                    column_config={
                        "Date": "Payment Date",
                        "Symbol": "Stock", 
                        "Dividend_Per_Share": "Per Share",
                        "Shares_Owned": "Shares",
                        "Total_Dividend_Received": "Amount Received",
                        "Quarter": "Quarter"
                    },
                    use_container_width=True,
                    hide_index=True
                )
            
            # Show recent payments (last 10) by default
            st.write("**🕒 Recent Dividend Payments (Last 10):**")
            recent_payments = display_history.head(10)
            st.dataframe(
                recent_payments[['Date', 'Symbol', 'Dividend_Per_Share', 'Total_Dividend_Received']],
                column_config={
                    "Date": "Payment Date",
                    "Symbol": "Stock",
                    "Dividend_Per_Share": "Per Share", 
                    "Total_Dividend_Received": "Amount Received"
                },
                use_container_width=True,
                hide_index=True
            )
            
            # Dividend frequency analysis
            st.subheader("📊 Dividend Payment Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Payment frequency by company
                payment_freq = dividend_history.groupby('Symbol')['Date'].count().reset_index()
                payment_freq.rename(columns={'Date': 'Payment_Count'}, inplace=True)
                avg_payments = payment_freq['Payment_Count'].mean()
                
                st.metric("Average Payments per Stock", f"{avg_payments:.1f}")
                
                # Most frequent payers
                frequent_payers = payment_freq.sort_values('Payment_Count', ascending=False).head(5)
                st.write("**Most Frequent Payers:**")
                for _, row in frequent_payers.iterrows():
                    st.write(f"• {row['Symbol']}: {row['Payment_Count']} payments")
            
            with col2:
                # Quarterly analysis
                quarterly_summary = dividend_history.groupby('Quarter')['Total_Dividend_Received'].sum().reset_index()
                quarterly_summary = quarterly_summary.sort_values('Total_Dividend_Received', ascending=False)
                
                if not quarterly_summary.empty:
                    best_quarter = quarterly_summary.iloc[0]
                    st.metric("Best Quarter", best_quarter['Quarter'], f"${best_quarter['Total_Dividend_Received']:.2f}")
                    
                    st.write("**Quarterly Performance:**")
                    for _, row in quarterly_summary.iterrows():
                        st.write(f"• {row['Quarter']}: ${row['Total_Dividend_Received']:.2f}")
            
            with col3:
                # Seasonality analysis
                dividend_history['Month_Num'] = pd.to_datetime(dividend_history['Date']).dt.month
                monthly_counts = dividend_history.groupby('Month_Num')['Total_Dividend_Received'].agg(['sum', 'count']).reset_index()
                monthly_counts['Month_Name'] = monthly_counts['Month_Num'].apply(lambda x: calendar.month_abbr[x])
                
                if not monthly_counts.empty:
                    best_month = monthly_counts.loc[monthly_counts['sum'].idxmax()]
                    st.metric("Best Month", best_month['Month_Name'], f"${best_month['sum']:.2f}")
                    
                    st.write("**Top Months:**")
                    top_months = monthly_counts.sort_values('sum', ascending=False).head(3)
                    for _, row in top_months.iterrows():
                        st.write(f"• {row['Month_Name']}: ${row['sum']:.2f}")
        
        else:
            st.warning("No dividend payments found in the last 12 months for this portfolio.")
            st.info("💡 This could mean:")
            st.write("• Your stocks don't pay dividends (growth stocks)")
            st.write("• Dividend payments are outside the 12-month window")
            st.write("• Data fetching issues with some stocks")
            st.write("• Try including some dividend-paying stocks (e.g., KO, JNJ, PG)"))
        
        # AI-Powered Analysis
        if include_ai_analysis:
            st.header("🤖 AI Portfolio Analysis")
            
            with st.spinner("Generating AI analysis..."):
                ai_analysis = analyzer.analyze_portfolio_with_ai(portfolio_df, metrics)
            
            st.markdown(ai_analysis)
        
        # Export options
        st.header("📥 Export & Backup Options")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Export portfolio summary
            summary_data = {
                'Portfolio Summary': [
                    f"Portfolio Name: {st.session_state.portfolio_name}",
                    f"Total Value: ${metrics['total_value']:,.2f}",
                    f"Annual Dividends: ${metrics['annual_dividends']:,.2f}",
                    f"Portfolio Yield: {metrics['portfolio_dividend_yield']:.2f}%",
                    f"Portfolio Beta: {metrics['portfolio_beta']:.2f}",
                    f"Number of Holdings: {len(portfolio_df)}",
                    f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            csv_summary = summary_df.to_csv(index=False)
            
            st.download_button(
                label="📊 Download Summary",
                data=csv_summary,
                file_name=f"{st.session_state.portfolio_name.replace(' ', '_')}_summary.csv",
                mime="text/csv"
            )
        
        with col2:
            # Export enhanced holdings CSV
            csv_holdings = analyzer.export_portfolio_csv(portfolio_df)
            
            st.download_button(
                label="📋 Download Holdings CSV",
                data=csv_holdings,
                file_name=f"{st.session_state.portfolio_name.replace(' ', '_')}_holdings.csv",
                mime="text/csv"
            )
        
        with col3:
            # Export complete portfolio backup (JSON)
            json_backup = analyzer.export_portfolio_json(
                portfolio_holdings, 
                metrics, 
                st.session_state.portfolio_name
            )
            
            st.download_button(
                label="💾 Download Full Backup",
                data=json_backup,
                file_name=f"{st.session_state.portfolio_name.replace(' ', '_')}_backup.json",
                mime="application/json",
                help="Complete portfolio backup with all settings and analysis"
            )
        
        with col4:
            # Export historical performance
            if not historical_performance.empty:
                csv_performance = historical_performance.to_csv()
                
                st.download_button(
                    label="📈 Download Performance",
                    data=csv_performance,
                    file_name=f"{st.session_state.portfolio_name.replace(' ', '_')}_performance.csv",
                    mime="text/csv"
                )
            else:
                st.button("📈 No Performance Data", disabled=True)
        
        # Additional export info
        st.info("""
        **💾 Export Options:**
        - **Summary**: Key metrics and overview
        - **Holdings CSV**: Current portfolio with market data  
        - **Full Backup**: Complete portfolio backup (restore with all settings)
        - **Performance**: Historical analysis data
        """)
    
    # Information section
    elif not portfolio_holdings:
        st.info("👈 Please add your portfolio holdings using the sidebar to begin analysis")
        
        # Show session restoration info if available
        if st.session_state.portfolio_holdings:
            st.warning(f"💡 You have a saved portfolio ({len(st.session_state.portfolio_holdings)} holdings). Use 'Load Saved Portfolio' to restore it.")
        
        st.markdown("---")
        st.subheader("ℹ️ About This Portfolio Analyzer")
        st.markdown("""
        This comprehensive tool provides:
        
        **📊 Core Analytics:**
        - Real-time portfolio valuation
        - Dividend income analysis and projections
        - Sector and geographic diversification
        - Risk metrics (Beta, concentration analysis)
        - Historical performance simulation
        
        **📅 Dividend Calendar:**
        - Monthly dividend distribution
        - Upcoming dividend payments
        - Dividend frequency analysis
        
        **🤖 AI-Powered Insights:**
        - Automated portfolio assessment
        - Diversification recommendations
        - Risk analysis and suggestions
        - Performance optimization tips
        
        **📈 Visualizations:**
        - Interactive charts and graphs
        - Sector allocation pie charts
        - Historical performance curves
        - Top holdings analysis
        
        **🔧 Input Methods:**
        - Manual entry of holdings
        - CSV file upload
        - Sample portfolio for testing
        """)
        
        st.markdown("---")
        st.markdown("**⚠️ Disclaimer:** This tool is for educational and analysis purposes only. Past performance does not guarantee future results. Always consult with financial advisors before making investment decisions.")

if __name__ == "__main__":
    main()
