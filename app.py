import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import json
import io
import time
import random
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configuration
st.set_page_config(
    page_title="Portfolio Optimizer", 
    page_icon="💼", 
    layout="wide"
)

class PortfolioAnalyzer:
    def __init__(self):
        self.init_session_state()
        
        # Sector mappings for common stocks
        self.sector_mapping = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'AMZN': 'Technology',
            'META': 'Technology', 'NVDA': 'Technology', 'ADBE': 'Technology', 'TSLA': 'Technology',
            'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare', 'ABBV': 'Healthcare',
            'LLY': 'Healthcare', 'TMO': 'Healthcare', 'ABT': 'Healthcare', 'MRK': 'Healthcare',
            'JPM': 'Financial Services', 'BAC': 'Financial Services', 'V': 'Financial Services',
            'WFC': 'Financial Services', 'GS': 'Financial Services', 'MS': 'Financial Services',
            'PG': 'Consumer Goods', 'KO': 'Consumer Goods', 'PEP': 'Consumer Goods',
            'WMT': 'Consumer Goods', 'HD': 'Consumer Goods', 'MCD': 'Consumer Goods',
            'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'EOG': 'Energy',
            'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities', 'AEP': 'Utilities',
            'CAT': 'Industrial', 'BA': 'Industrial', 'GE': 'Industrial', 'MMM': 'Industrial',
            'AMT': 'Real Estate', 'PLD': 'Real Estate', 'CCI': 'Real Estate',
            'T': 'Communication Services', 'VZ': 'Communication Services', 'DIS': 'Communication Services'
        }

    def init_session_state(self):
        """Initialize session state"""
        if 'portfolio_holdings' not in st.session_state:
            st.session_state.portfolio_holdings = []
        if 'portfolio_name' not in st.session_state:
            st.session_state.portfolio_name = "My Portfolio"

    def fetch_stock_info(self, symbol: str) -> Dict:
        """Fetch stock information with rate limiting"""
        time.sleep(1.2)  # Rate limiting for Yahoo Finance
        
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            if not info or len(info) < 5:
                raise Exception("Empty data returned")
            
            # Handle dividend yield properly
            dividend_yield_raw = info.get('dividendYield', 0)
            dividend_yield = dividend_yield_raw * 100 if dividend_yield_raw and dividend_yield_raw < 1 else dividend_yield_raw or 0
            
            # Handle payout ratio
            payout_ratio_raw = info.get('payoutRatio', 0)
            payout_ratio = payout_ratio_raw * 100 if payout_ratio_raw and payout_ratio_raw < 1 else payout_ratio_raw or 0
            
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
                'country': info.get('country', 'Unknown')
            }
                
        except Exception as e:
            st.warning(f"Error fetching data for {symbol}: {str(e)}")
            return {
                'symbol': symbol, 'name': symbol, 'current_price': 0, 'dividend_yield': 0,
                'dividend_rate': 0, 'payout_ratio': 0, 'pe_ratio': 0, 'market_cap': 0,
                'beta': 1, 'sector': self.sector_mapping.get(symbol, 'Unknown'), 
                'industry': 'Unknown', 'country': 'Unknown'
            }

    def calculate_portfolio_metrics(self, portfolio_data: pd.DataFrame) -> Dict:
        """Calculate portfolio metrics"""
        metrics = {}
        
        # Current portfolio value
        total_value = (portfolio_data['shares'] * portfolio_data['current_price']).sum()
        metrics['total_value'] = total_value
        
        if total_value == 0:
            return metrics
        
        # Portfolio weights
        portfolio_data['weight'] = (portfolio_data['shares'] * portfolio_data['current_price']) / total_value
        
        # Annual dividend income
        annual_dividends = (portfolio_data['shares'] * portfolio_data['dividend_rate']).sum()
        metrics['annual_dividends'] = annual_dividends
        
        # Portfolio dividend yield
        metrics['portfolio_dividend_yield'] = (annual_dividends / total_value) * 100
        
        # Weighted P/E ratio
        portfolio_data['pe_weighted'] = portfolio_data['weight'] * portfolio_data['pe_ratio']
        metrics['weighted_pe'] = portfolio_data['pe_weighted'].sum()
        
        # Portfolio beta
        portfolio_data['beta_weighted'] = portfolio_data['weight'] * portfolio_data['beta']
        metrics['portfolio_beta'] = portfolio_data['beta_weighted'].sum()
        
        # Sector diversification
        sector_allocation = portfolio_data.groupby('sector').agg({
            'weight': 'sum'
        })
        metrics['sector_allocation'] = sector_allocation
        
        # Top holdings
        portfolio_data_sorted = portfolio_data.sort_values('weight', ascending=False)
        metrics['top_holdings'] = portfolio_data_sorted.head(10)
        
        return metrics

    def calculate_synthesis_metrics(self, portfolio_data: pd.DataFrame, metrics: Dict) -> Dict:
        """Calculate synthesis metrics for optimization table"""
        if metrics.get('total_value', 0) == 0:
            return {
                'sharpe_ratio': 0,
                'mean_opportunity_margin': 0,
                'annualized_total_return': 0,
                'sector_diversification_score': 0
            }
        
        synthesis = {}
        
        # 1. Sharpe Ratio (estimated)
        risk_free_rate = 0.04
        estimated_return = metrics['portfolio_dividend_yield'] / 100 + 0.08  # Div yield + estimated growth
        estimated_volatility = metrics['portfolio_beta'] * 0.16  # Market vol * portfolio beta
        synthesis['sharpe_ratio'] = (estimated_return - risk_free_rate) / estimated_volatility if estimated_volatility > 0 else 0
        
        # 2. Mean Opportunity Margin (P/E based valuation)
        valid_pe_data = portfolio_data[portfolio_data['pe_ratio'] > 0]
        if len(valid_pe_data) > 0:
            portfolio_weights = valid_pe_data['shares'] * valid_pe_data['current_price']
            portfolio_weights = portfolio_weights / portfolio_weights.sum()
            weighted_pe = (valid_pe_data['pe_ratio'] * portfolio_weights).sum()
            market_pe = 20  # Market average
            synthesis['mean_opportunity_margin'] = ((market_pe - weighted_pe) / market_pe) * 100
        else:
            synthesis['mean_opportunity_margin'] = 0
        
        # 3. Annualized Total Return (estimated)
        dividend_yield = metrics['portfolio_dividend_yield']
        estimated_price_appreciation = 8  # 8% estimated
        synthesis['annualized_total_return'] = dividend_yield + estimated_price_appreciation
        
        # 4. Sector Diversification Score (explained)
        sector_allocation = metrics['sector_allocation']
        num_sectors = len(sector_allocation)
        
        # Explanation: Good diversification means:
        # 1. Multiple sectors (more is better)
        # 2. No single sector dominates (< 25% each)
        # 3. Balanced distribution across sectors
        
        # Points for number of sectors (0-50 points)
        sector_points = min(50, num_sectors * 8)  # 8 points per sector, max 50
        
        # Penalty for concentration (0-50 penalty)
        max_sector_weight = sector_allocation['weight'].max() if len(sector_allocation) > 0 else 0
        concentration_penalty = max(0, (max_sector_weight - 0.25) * 100)  # Penalty if >25% in one sector
        
        # Balance score (0-30 points)
        if num_sectors > 0:
            ideal_weight = 1.0 / num_sectors  # Perfect balance
            deviations = abs(sector_allocation['weight'] - ideal_weight)
            balance_score = max(0, 30 - deviations.sum() * 100)
        else:
            balance_score = 0
        
        # Total: sector_points (0-50) - concentration_penalty (0-50) + balance_score (0-30) = 0-100
        synthesis['sector_diversification_score'] = max(0, min(100, sector_points - concentration_penalty + balance_score))
        
        return synthesis

    def calculate_synthesis_metrics_basic(self, portfolio_data: pd.DataFrame, metrics: Dict) -> Dict:
        """Calculate synthesis metrics without historical performance (fallback)"""
        if metrics.get('total_value', 0) == 0:
            return {
                'sharpe_ratio': 0,
                'mean_opportunity_margin': 0,
                'annualized_total_return': 0,
                'sector_diversification_score': 0,
                'data_source': 'insufficient_data'
            }
        
        synthesis = {}
        
        # 1. Sharpe Ratio (estimated)
        risk_free_rate = 0.04
        estimated_return = metrics.get('portfolio_dividend_yield', 0) / 100 + 0.08
        estimated_volatility = metrics.get('portfolio_beta', 1) * 0.16
        synthesis['sharpe_ratio'] = (estimated_return - risk_free_rate) / estimated_volatility if estimated_volatility > 0 else 0
        
        # 2. Mean Opportunity Margin
        valid_pe_data = portfolio_data[portfolio_data['pe_ratio'] > 0]
        if len(valid_pe_data) > 0:
            portfolio_weights = valid_pe_data['shares'] * valid_pe_data['current_price']
            portfolio_weights = portfolio_weights / portfolio_weights.sum()
            weighted_pe = (valid_pe_data['pe_ratio'] * portfolio_weights).sum()
            market_pe = 20
            synthesis['mean_opportunity_margin'] = ((market_pe - weighted_pe) / market_pe) * 100
        else:
            synthesis['mean_opportunity_margin'] = 0
        
        # 3. Annualized Total Return (estimated only)
        dividend_yield = metrics.get('portfolio_dividend_yield', 0)
        estimated_price_appreciation = 8
        synthesis['annualized_total_return'] = dividend_yield + estimated_price_appreciation
        synthesis['data_source'] = 'estimated'
        
        # 4. Sector Diversification Score
        try:
            sector_allocation = metrics.get('sector_allocation', pd.DataFrame())
            num_sectors = len(sector_allocation)
            
            sector_points = min(50, num_sectors * 8)
            
            if len(sector_allocation) > 0:
                max_sector_weight = sector_allocation['weight'].max()
                concentration_penalty = max(0, (max_sector_weight - 0.25) * 100)
            else:
                max_sector_weight = 0
                concentration_penalty = 0
            
            if num_sectors > 0:
                ideal_weight = 1.0 / num_sectors
                deviations = abs(sector_allocation['weight'] - ideal_weight)
                balance_score = max(0, 30 - deviations.sum() * 100)
            else:
                balance_score = 0
            
            final_score = sector_points - concentration_penalty + balance_score
            synthesis['sector_diversification_score'] = max(0, min(100, final_score))
            
            synthesis['diversification_breakdown'] = {
                'num_sectors': num_sectors,
                'sector_points': sector_points,
                'max_sector_weight': max_sector_weight * 100,
                'concentration_penalty': concentration_penalty,
                'balance_score': balance_score,
                'final_score': final_score
            }
            
        except Exception:
            synthesis['sector_diversification_score'] = 0
            synthesis['diversification_breakdown'] = {
                'num_sectors': 0,
                'sector_points': 0,
                'max_sector_weight': 0,
                'concentration_penalty': 0,
                'balance_score': 0,
                'final_score': 0
            }
        
        return synthesis

    def simulate_historical_performance(self, portfolio_holdings: List[Dict], years: int = 3, show_dividend_details: bool = False) -> pd.DataFrame:
        """Simulate historical portfolio performance with total return (price + dividends)"""
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years*365)
        
        # Fetch historical data for all holdings
        all_data = {}
        total_initial_value = 0
        successful_fetches = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, holding in enumerate(portfolio_holdings):
            symbol = holding['symbol']
            shares = holding['shares']
            
            status_text.text(f'Fetching {years}-year historical data for {symbol}... ({i+1}/{len(portfolio_holdings)})')
            progress_bar.progress((i + 1) / len(portfolio_holdings))
            
            # Skip if no shares
            if shares <= 0:
                continue
            
            try:
                # 1-second delay for Yahoo Finance
                time.sleep(1.0)
                
                stock = yf.Ticker(symbol)
                
                # Get historical data
                hist = stock.history(start=start_date, end=end_date, auto_adjust=True, back_adjust=True)
                
                if hist.empty or len(hist) < 10:
                    st.warning(f"Insufficient historical data for {symbol}")
                    continue
                
                # Calculate position value over time
                hist['Position_Value'] = hist['Close'] * shares
                
                # Get dividend data
                total_dividend_income = 0
                dividend_payments_found = 0
                
                try:
                    dividends = stock.dividends
                    hist['Dividend_Income'] = 0
                    
                    if len(dividends) > 0:
                        if show_dividend_details:
                            st.info(f"📊 Found {len(dividends)} dividend payments for {symbol}")
                        
                        # Add dividends received on each date
                        for div_date, div_amount in dividends.items():
                            try:
                                # Handle timezone issues
                                if hasattr(div_date, 'tz') and div_date.tz is not None:
                                    div_date_normalized = div_date.tz_localize(None)
                                else:
                                    div_date_normalized = div_date
                                
                                div_date_only = div_date_normalized.date()
                                hist_dates = [d.date() for d in hist.index]
                                
                                # Look for exact match or closest date within 3 days
                                if div_date_only in hist_dates:
                                    match_date = next(d for d in hist.index if d.date() == div_date_only)
                                    dividend_amount = float(div_amount) * shares
                                    hist.loc[match_date, 'Dividend_Income'] += dividend_amount
                                    total_dividend_income += dividend_amount
                                    dividend_payments_found += 1
                                else:
                                    # Look for closest date within 3 days
                                    closest_date = None
                                    min_diff = float('inf')
                                    
                                    for hist_date in hist.index:
                                        date_diff = abs((hist_date.date() - div_date_only).days)
                                        if date_diff <= 3 and date_diff < min_diff:
                                            min_diff = date_diff
                                            closest_date = hist_date
                                    
                                    if closest_date is not None:
                                        dividend_amount = float(div_amount) * shares
                                        hist.loc[closest_date, 'Dividend_Income'] += dividend_amount
                                        total_dividend_income += dividend_amount
                                        dividend_payments_found += 1
                                        
                            except Exception as div_error:
                                if show_dividend_details:
                                    st.warning(f"Error processing dividend for {symbol} on {div_date}: {str(div_error)}")
                                continue
                    
                    if show_dividend_details:
                        if total_dividend_income > 0:
                            st.success(f"✅ {symbol}: ${total_dividend_income:.2f} total dividends ({dividend_payments_found} payments)")
                        else:
                            st.info(f"ℹ️ No dividend payments found for {symbol}")
                        
                except Exception as e:
                    if show_dividend_details:
                        st.warning(f"Could not fetch dividend data for {symbol}: {str(e)}")
                    hist['Dividend_Income'] = 0
                
                # Calculate cumulative dividends
                hist['Cumulative_Dividends'] = hist['Dividend_Income'].cumsum()
                
                # Calculate total return percentage (capital gains + dividends)
                initial_price = hist['Close'].iloc[0]
                if initial_price <= 0:
                    st.warning(f"Invalid initial price for {symbol}: {initial_price}")
                    continue
                    
                initial_investment = initial_price * shares
                
                # Total value = current position value + all dividends received
                hist['Total_Value'] = hist['Position_Value'] + hist['Cumulative_Dividends']
                
                # Total return percentage
                hist['Total_Return_Pct'] = ((hist['Total_Value'] / initial_investment) - 1) * 100
                
                # Price-only return
                hist['Price_Return_Pct'] = ((hist['Close'] / initial_price) - 1) * 100
                
                all_data[symbol] = hist
                total_initial_value += initial_investment
                successful_fetches += 1
                
            except Exception as e:
                st.warning(f"Error processing historical data for {symbol}: {str(e)}")
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        # Check if we have enough data
        if len(all_data) == 0:
            st.error("❌ No historical data could be fetched for any stocks in your portfolio.")
            return pd.DataFrame()
        
        if successful_fetches < len(portfolio_holdings) * 0.5:
            st.warning(f"⚠️ Only {successful_fetches}/{len(portfolio_holdings)} stocks had historical data fetched.")
        
        # Combine all positions into portfolio performance
        portfolio_performance = pd.DataFrame()
        
        # Normalize dates to handle timezone issues
        normalized_data = {}
        for symbol, data in all_data.items():
            data_copy = data.copy()
            data_copy.index = data_copy.index.normalize().date
            normalized_data[symbol] = data_copy
        
        # Find common dates
        date_coverage = {}
        for symbol, data in normalized_data.items():
            date_coverage[symbol] = set(data.index)
        
        all_dates = set()
        for dates in date_coverage.values():
            all_dates.update(dates)
        
        # Count stocks per date
        date_counts = {}
        for date in all_dates:
            count = sum(1 for stock_dates in date_coverage.values() if date in stock_dates)
            date_counts[date] = count
        
        # Use dates where at least 70% of stocks have data
        min_stocks_required = max(1, int(len(all_data) * 0.7))
        common_dates = [date for date, count in date_counts.items() if count >= min_stocks_required]
        common_dates = sorted(common_dates)
        
        if len(common_dates) == 0:
            st.error("❌ No overlapping dates found across stocks.")
            return pd.DataFrame()
        
        # Calculate portfolio metrics for each date
        for date in common_dates:
            portfolio_value = 0
            total_dividends = 0
            stocks_with_data = 0
            
            for symbol, data in normalized_data.items():
                if date in data.index:
                    portfolio_value += data.loc[date, 'Position_Value']
                    total_dividends += data.loc[date, 'Cumulative_Dividends']
                    stocks_with_data += 1
            
            if stocks_with_data >= min_stocks_required:
                date_timestamp = pd.Timestamp(date)
                portfolio_performance.loc[date_timestamp, 'Portfolio_Value'] = portfolio_value
                portfolio_performance.loc[date_timestamp, 'Total_Dividends'] = total_dividends
                portfolio_performance.loc[date_timestamp, 'Total_Value'] = portfolio_value + total_dividends
                portfolio_performance.loc[date_timestamp, 'Stocks_With_Data'] = stocks_with_data
                
                # Total return including dividends
                if total_initial_value > 0:
                    portfolio_performance.loc[date_timestamp, 'Total_Return'] = ((portfolio_value + total_dividends) / total_initial_value - 1) * 100
                    portfolio_performance.loc[date_timestamp, 'Price_Only_Return'] = (portfolio_value / total_initial_value - 1) * 100
                else:
                    portfolio_performance.loc[date_timestamp, 'Total_Return'] = 0
                    portfolio_performance.loc[date_timestamp, 'Price_Only_Return'] = 0
        
        # Success message
        actual_days = len(portfolio_performance)
        total_portfolio_dividends = portfolio_performance['Total_Dividends'].iloc[-1] if not portfolio_performance.empty else 0
        
        st.success(f"✅ Historical analysis complete! {successful_fetches} stocks analyzed over {actual_days} trading days")
        
        if total_portfolio_dividends > 0:
            st.info(f"💰 Total dividends captured: ${total_portfolio_dividends:,.2f} over {years} years")
        
        return portfolio_performance.sort_index()

    def generate_historical_dividend_calendar(self, portfolio_holdings: List[Dict]) -> pd.DataFrame:
        """Generate historical dividend calendar showing actual payments from last year"""
        dividend_history = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Get date range for last year
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        for i, holding in enumerate(portfolio_holdings):
            symbol = holding['symbol']
            shares = holding['shares']
            
            status_text.text(f'Fetching dividend history for {symbol}... ({i+1}/{len(portfolio_holdings)})')
            progress_bar.progress((i + 1) / len(portfolio_holdings))
            
            try:
                # 1-second delay for Yahoo Finance
                time.sleep(1.0)
                
                stock = yf.Ticker(symbol)
                
                # Get dividend history for the last year
                dividends = stock.dividends
                
                # Filter dividends from last year
                if len(dividends) > 0:
                    dividend_dates = dividends.index.tz_localize(None) if dividends.index.tz is not None else dividends.index
                    
                    recent_mask = (dividend_dates >= start_date) & (dividend_dates <= end_date)
                    recent_dividends = dividends[recent_mask]
                    recent_dividend_dates = dividend_dates[recent_mask]
                    
                    # Add each dividend payment to history
                    for date, dividend_amount in zip(recent_dividend_dates, recent_dividends):
                        dividend_history.append({
                            'Date': date.date(),
                            'Symbol': symbol,
                            'Company': symbol,
                            'Dividend_Per_Share': float(dividend_amount),
                            'Shares_Owned': shares,
                            'Total_Dividend_Received': float(dividend_amount) * shares,
                            'Month': date.strftime('%Y-%m'),
                            'Quarter': f"Q{((date.month-1)//3)+1} {date.year}"
                        })
            
            except Exception as e:
                st.warning(f"Could not fetch dividend history for {symbol}: {str(e)}")
        
        progress_bar.empty()
        status_text.empty()
        
        if dividend_history:
            df = pd.DataFrame(dividend_history)
            df = df.sort_values('Date')
            
            # Create daily aggregation
            daily_dividends = df.groupby('Date').agg({
                'Total_Dividend_Received': 'sum',
                'Symbol': lambda x: ', '.join(x.unique())
            }).reset_index()
            daily_dividends.rename(columns={'Symbol': 'Companies'}, inplace=True)
            
            # Create monthly aggregation
            monthly_dividends = df.groupby('Month').agg({
                'Total_Dividend_Received': 'sum',
                'Symbol': lambda x: len(x.unique())
            }).reset_index()
            monthly_dividends.rename(columns={'Symbol': 'Companies_Count'}, inplace=True)
            
            return df, daily_dividends, monthly_dividends
        else:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def import_portfolio_from_csv(self, csv_data: str) -> List[Dict]:
        """Import portfolio from CSV"""
        try:
            df = pd.read_csv(io.StringIO(csv_data))
            
            if 'symbol' not in df.columns or 'shares' not in df.columns:
                raise ValueError("CSV must contain 'symbol' and 'shares' columns")
            
            holdings = []
            for _, row in df.iterrows():
                holdings.append({
                    'symbol': str(row['symbol']).upper().strip(),
                    'shares': float(row['shares'])
                })
            
            return holdings
            
        except Exception as e:
            raise ValueError(f"Error importing CSV: {str(e)}")

def main():
    st.title("💼 Portfolio Optimizer")
    st.markdown("**Simple portfolio optimization with synthesis metrics**")
    
    analyzer = PortfolioAnalyzer()
    
    # Simplified sidebar
    st.sidebar.header("📁 Portfolio Management")
    
    # Load CSV
    st.sidebar.subheader("Upload Portfolio CSV")
    uploaded_file = st.sidebar.file_uploader(
        "Choose CSV file", 
        type="csv",
        help="CSV should have columns: symbol, shares"
    )
    
    # Initialize portfolio holdings
    if 'holdings' not in st.session_state:
        st.session_state.holdings = []
    
    # CSV Upload handling
    if uploaded_file:
        try:
            csv_content = uploaded_file.read().decode('utf-8')
            imported_holdings = analyzer.import_portfolio_from_csv(csv_content)
            st.session_state.holdings = imported_holdings.copy()
            st.sidebar.success(f"✅ Loaded {len(imported_holdings)} holdings")
        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")
    
    # Add stocks section (always available)
    st.sidebar.subheader("➕ Add/Modify Stocks")
    
    with st.sidebar.form("add_stock_form"):
        new_symbol = st.text_input("Stock Symbol", placeholder="e.g., AAPL").upper()
        
        allocation_method = st.radio(
            "How to add:",
            ["Equal Weight (Auto)", "Custom Shares", "Custom Weight %"]
        )
        
        if allocation_method == "Custom Shares":
            new_shares = st.number_input("Number of Shares", min_value=0.0, value=100.0, step=1.0)
        elif allocation_method == "Custom Weight %":
            target_weight = st.number_input("Target Weight %", min_value=1.0, max_value=50.0, value=10.0)
        else:
            new_shares = None
            target_weight = None
        
        if st.form_submit_button("Add/Update Stock"):
            if new_symbol:
                # Remove if exists
                st.session_state.holdings = [h for h in st.session_state.holdings if h['symbol'] != new_symbol]
                
                if allocation_method == "Custom Shares":
                    st.session_state.holdings.append({'symbol': new_symbol, 'shares': new_shares})
                    st.sidebar.success(f"Added {new_symbol}: {new_shares} shares")
                
                elif allocation_method == "Custom Weight %":
                    # Calculate shares based on weight (estimated)
                    estimated_portfolio_value = 100000  # Default estimate
                    estimated_price = 150  # Default stock price estimate
                    calculated_shares = (estimated_portfolio_value * target_weight / 100) / estimated_price
                    st.session_state.holdings.append({
                        'symbol': new_symbol, 
                        'shares': max(1, int(calculated_shares)),
                        'target_weight': target_weight / 100,
                        'allocation_method': 'weight'
                    })
                    st.sidebar.success(f"Added {new_symbol}: {target_weight}% target weight")
                
                else:  # Equal Weight
                    if len(st.session_state.holdings) > 0:
                        # Equal weight among all stocks
                        target_weight = 1.0 / (len(st.session_state.holdings) + 1)
                        estimated_portfolio_value = 100000
                        estimated_price = 150
                        calculated_shares = (estimated_portfolio_value * target_weight) / estimated_price
                        st.session_state.holdings.append({
                            'symbol': new_symbol, 
                            'shares': max(1, int(calculated_shares)),
                            'target_weight': target_weight,
                            'allocation_method': 'equal'
                        })
                        st.sidebar.success(f"Added {new_symbol}: Equal weight ({target_weight*100:.1f}%)")
                    else:
                        st.session_state.holdings.append({'symbol': new_symbol, 'shares': 100})
                        st.sidebar.success(f"Added {new_symbol}: 100 shares (first stock)")
    
    # Remove stocks section
    if st.session_state.holdings:
        st.sidebar.subheader("🗑️ Remove Stocks")
        symbols_to_remove = st.sidebar.multiselect(
            "Select stocks to remove:",
            [h['symbol'] for h in st.session_state.holdings]
        )
        
        if st.sidebar.button("Remove Selected") and symbols_to_remove:
            st.session_state.holdings = [h for h in st.session_state.holdings if h['symbol'] not in symbols_to_remove]
            st.sidebar.success(f"Removed {len(symbols_to_remove)} stocks")
            st.rerun()
    
    # Current portfolio display
    if st.session_state.holdings:
        st.sidebar.subheader("📊 Current Portfolio")
        for holding in st.session_state.holdings:
            if holding.get('allocation_method') in ['weight', 'equal']:
                target_weight = holding.get('target_weight', 0) * 100
                st.sidebar.write(f"• {holding['symbol']}: {holding['shares']} shares ({target_weight:.1f}% target)")
            else:
                st.sidebar.write(f"• {holding['symbol']}: {holding['shares']} shares")
        
        st.sidebar.info(f"Total: {len(st.session_state.holdings)} stocks")
    
    # Analysis settings
    st.sidebar.subheader("⚙️ Settings")
    analysis_period = st.sidebar.selectbox("Historical Period", ["1y", "2y", "3y", "5y"], index=2)
    show_dividend_details = st.sidebar.checkbox("Show Dividend Details", value=False)
    
    # MAIN ANALYSIS
    portfolio_holdings = st.session_state.holdings
    
    if portfolio_holdings and st.sidebar.button("🚀 Analyze Portfolio", type="primary"):
        
        st.info(f"Analyzing {len(portfolio_holdings)} holdings...")
        
        # Fetch data
        portfolio_data = []
        progress_bar = st.progress(0)
        
        for i, holding in enumerate(portfolio_holdings):
            symbol = holding['symbol']
            progress_bar.progress((i + 1) / len(portfolio_holdings))
            
            stock_info = analyzer.fetch_stock_info(symbol)
            
            # Handle proportional allocation
            if holding.get('allocation_method') in ['weight', 'equal']:
                # Recalculate shares based on actual price and total portfolio value
                # For now, use the estimated shares from input
                stock_info['shares'] = holding['shares']
            else:
                stock_info['shares'] = holding['shares']
            
            portfolio_data.append(stock_info)
        
        progress_bar.empty()
        
        portfolio_df = pd.DataFrame(portfolio_data)
        
        # Calculate basic metrics first
        metrics = analyzer.calculate_portfolio_metrics(portfolio_df)
        
        # Calculate historical performance FIRST
        st.info("📊 Calculating historical performance...")
        years = int(analysis_period[0]) if analysis_period[0].isdigit() else 3
        
        with st.spinner("Calculating historical performance..."):
            historical_performance = analyzer.simulate_historical_performance(
                portfolio_holdings, 
                years=years,
                show_dividend_details=show_dividend_details
            )
        
        # Portfolio Overview
        st.header("📊 Portfolio Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Value", f"${metrics['total_value']:,.2f}")
        with col2:
            st.metric("Annual Dividends", f"${metrics['annual_dividends']:,.2f}")
        with col3:
            st.metric("Dividend Yield", f"{metrics['portfolio_dividend_yield']:.2f}%")
        with col4:
            st.metric("Portfolio Beta", f"{metrics['portfolio_beta']:.2f}")
        
        # Holdings table
        st.subheader("📋 Holdings")
        
        display_df = portfolio_df.copy()
        display_df['Value'] = display_df['shares'] * display_df['current_price']
        display_df['Weight %'] = (display_df['Value'] / metrics['total_value']) * 100
        display_df['Annual Div'] = display_df['shares'] * display_df['dividend_rate']
        
        st.dataframe(
            display_df[['symbol', 'name', 'sector', 'shares', 'current_price', 'Value', 'Weight %', 'dividend_yield', 'Annual Div']],
            column_config={
                "symbol": "Symbol",
                "name": "Company",
                "sector": "Sector", 
                "shares": "Shares",
                "current_price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "Value": st.column_config.NumberColumn("Value", format="$%.2f"),
                "Weight %": st.column_config.NumberColumn("Weight %", format="%.1f%%"),
                "dividend_yield": st.column_config.NumberColumn("Div Yield %", format="%.2f%%"),
                "Annual Div": st.column_config.NumberColumn("Annual Div", format="$%.2f")
            },
            use_container_width=True,
            hide_index=True
        )
        
        # Visualizations
        st.subheader("📈 Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sector allocation
            sector_data = metrics['sector_allocation'].reset_index()
            fig_sector = px.pie(
                sector_data, 
                values='weight', 
                names='sector',
                title="Sector Allocation"
            )
            st.plotly_chart(fig_sector, use_container_width=True)
        
        with col2:
            # Top holdings
            top_holdings = metrics['top_holdings'].head(8)
            fig_holdings = px.bar(
                top_holdings,
                x='weight',
                y='symbol',
                orientation='h',
                title="Top Holdings by Weight"
            )
            fig_holdings.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_holdings, use_container_width=True)
        
        # Historical Performance Analysis
        st.header("📈 Historical Performance Analysis")
        st.info("💡 **Total Return** includes both price appreciation AND dividend reinvestment")
        
        with st.spinner("Calculating historical performance..."):
            years = int(analysis_period[0]) if analysis_period[0].isdigit() else 3
            historical_performance = analyzer.simulate_historical_performance(
                portfolio_holdings, 
                years=years,
                show_dividend_details=show_dividend_details
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
                    st.metric("Total Return", f"{total_return:.1f}%", help="Includes price appreciation + dividends")
                with col2:
                    st.metric("Annualized Return", f"{annual_total_return*100:.1f}%", help="Compound annual growth rate")
                with col3:
                    final_dividends = historical_performance['Total_Dividends'].iloc[-1]
                    st.metric("Total Dividends", f"${final_dividends:,.0f}", help="Cumulative dividends received")
                with col4:
                    st.metric("Dividend Contribution", f"{dividend_contribution:.1f}%", help="Return boost from dividends")
                
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
        
        dividend_history, daily_dividends, monthly_dividends = analyzer.generate_historical_dividend_calendar(portfolio_holdings)
        
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
                    fig_daily = go.Figure()
                    
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
                        height=400,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_daily, use_container_width=True)
                
                # Monthly aggregation chart
                if not monthly_dividends.empty:
                    st.subheader("📈 Monthly Dividend Summary")
                    
                    fig_monthly = go.Figure()
                    
                    fig_monthly.add_trace(go.Bar(
                        x=monthly_dividends['Month'],
                        y=monthly_dividends['Total_Dividend_Received'],
                        text=monthly_dividends['Total_Dividend_Received'].apply(lambda x: f'${x:.0f}'),
                        textposition='outside',
                        marker_color='darkgreen',
                        name='Monthly Dividends'
                    ))
                    
                    fig_monthly.update_layout(
                        title="Monthly Dividend Income",
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
                    'Dividend_Per_Share': 'count',
                    'Date': 'max'
                }).reset_index()
                company_summary.rename(columns={
                    'Dividend_Per_Share': 'Payment_Count',
                    'Date': 'Last_Payment'
                }, inplace=True)
                company_summary = company_summary.sort_values('Total_Dividend_Received', ascending=False)
                
                # Company pie chart
                fig_pie = px.pie(
                    company_summary.head(8),
                    values='Total_Dividend_Received',
                    names='Symbol',
                    title="Dividend Distribution by Company"
                )
                fig_pie.update_layout(height=300)
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # Top dividend payers table
                st.subheader("🏆 Top Dividend Payers")
                top_payers = company_summary.head(8).copy()
                top_payers['Total_Dividend_Received'] = top_payers['Total_Dividend_Received'].apply(lambda x: f"${x:.2f}")
                top_payers['Last_Payment'] = top_payers['Last_Payment'].apply(lambda x: x.strftime('%Y-%m-%d'))
                
                st.dataframe(
                    top_payers[['Symbol', 'Total_Dividend_Received', 'Payment_Count']],
                    column_config={
                        "Symbol": "Stock",
                        "Total_Dividend_Received": "Total",
                        "Payment_Count": "Payments"
                    },
                    use_container_width=True,
                    hide_index=True
                )
            
            # Recent payments table
            st.subheader("🕒 Recent Dividend Payments")
            display_history = dividend_history.copy()
            display_history['Date'] = display_history['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
            display_history['Dividend_Per_Share'] = display_history['Dividend_Per_Share'].apply(lambda x: f"${x:.4f}")
            display_history['Total_Dividend_Received'] = display_history['Total_Dividend_Received'].apply(lambda x: f"${x:.2f}")
            display_history = display_history.sort_values('Date', ascending=False)
            
            recent_payments = display_history.head(10)
            st.dataframe(
                recent_payments[['Date', 'Symbol', 'Dividend_Per_Share', 'Total_Dividend_Received']],
                column_config={
                    "Date": "Date",
                    "Symbol": "Stock",
                    "Dividend_Per_Share": "Per Share", 
                    "Total_Dividend_Received": "Amount"
                },
                use_container_width=True,
                hide_index=True
            )
        
        else:
            st.warning("No dividend payments found in the last 12 months for this portfolio.")
            st.info("💡 This could mean:")
            st.write("• Your stocks don't pay dividends (growth stocks)")
            st.write("• Dividend payments are outside the 12-month window")
            st.write("• Try including dividend-paying stocks (e.g., KO, JNJ, PG)")
        
        # 🎯 SYNTHESIS TABLE (AT THE END WITH ACTUAL DATA)
        st.header("🎯 Portfolio Synthesis Metrics")
        st.info("📊 **Final synthesis using actual calculated values from analysis above**")
        
        # Get actual annualized return from historical performance if available
        actual_annual_return_for_synthesis = None
        if not historical_performance.empty and len(historical_performance) > 1:
            total_return_for_synthesis = historical_performance['Total_Return'].iloc[-1]
            actual_annual_return_for_synthesis = ((1 + total_return_for_synthesis/100) ** (1/years) - 1) * 100
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            # Calculate Sharpe ratio
            risk_free_rate = 0.04
            estimated_return = metrics['portfolio_dividend_yield'] / 100 + 0.08
            estimated_volatility = metrics['portfolio_beta'] * 0.16
            sharpe_ratio = (estimated_return - risk_free_rate) / estimated_volatility if estimated_volatility > 0 else 0
            
            sharpe_color = "🟢" if sharpe_ratio > 1.0 else "🟡" if sharpe_ratio > 0.5 else "🔴"
            st.metric(
                "Sharpe Coefficient", 
                f"{sharpe_color} {sharpe_ratio:.2f}",
                help="Risk-adjusted return (>1.0 = good, >2.0 = excellent)"
            )
        
        with col2:
            # Calculate opportunity margin
            valid_pe_data = portfolio_df[portfolio_df['pe_ratio'] > 0]
            if len(valid_pe_data) > 0:
                portfolio_weights = valid_pe_data['shares'] * valid_pe_data['current_price']
                portfolio_weights = portfolio_weights / portfolio_weights.sum()
                weighted_pe = (valid_pe_data['pe_ratio'] * portfolio_weights).sum()
                market_pe = 20
                opportunity_margin = ((market_pe - weighted_pe) / market_pe) * 100
            else:
                opportunity_margin = 0
            
            margin_color = "🟢" if opportunity_margin > 10 else "🟡" if opportunity_margin > -10 else "🔴"
            st.metric(
                "Mean Opp. Margin", 
                f"{margin_color} {opportunity_margin:+.1f}%",
                help="Valuation vs market (+% = undervalued)"
            )
        
        with col3:
            # Use ACTUAL annualized return from curves or estimate
            if actual_annual_return_for_synthesis is not None:
                annual_return_display = actual_annual_return_for_synthesis
                data_source_icon = "📊"
                help_text = "Actual annual return from historical performance curves above"
            else:
                annual_return_display = metrics['portfolio_dividend_yield'] + 8  # Estimate
                data_source_icon = "📈"
                help_text = "Estimated (no historical data available)"
            
            return_color = "🟢" if annual_return_display > 12 else "🟡" if annual_return_display > 8 else "🔴"
            st.metric(
                "Ann. Total Return (Div)", 
                f"{return_color} {annual_return_display:.1f}%",
                help=f"{data_source_icon} {help_text}"
            )
        
        with col4:
            # Dividend Yield
            div_yield = metrics['portfolio_dividend_yield']
            yield_color = "🟢" if div_yield > 4 else "🟡" if div_yield > 2 else "🔴"
            st.metric(
                "Dividend Yield", 
                f"{yield_color} {div_yield:.1f}%",
                help="Current annual dividend yield"
            )
        
        with col5:
            # Calculate sector diversification score
            sector_allocation = metrics['sector_allocation']
            num_sectors = len(sector_allocation)
            sector_points = min(50, num_sectors * 8)
            
            if len(sector_allocation) > 0:
                max_sector_weight = sector_allocation['weight'].max()
                concentration_penalty = max(0, (max_sector_weight - 0.25) * 100)
            else:
                max_sector_weight = 0
                concentration_penalty = 0
            
            if num_sectors > 0:
                ideal_weight = 1.0 / num_sectors
                deviations = abs(sector_allocation['weight'] - ideal_weight)
                balance_score = max(0, 30 - deviations.sum() * 100)
            else:
                balance_score = 0
            
            diversification_score = max(0, min(100, sector_points - concentration_penalty + balance_score))
            
            div_color = "🟢" if diversification_score > 70 else "🟡" if diversification_score > 40 else "🔴"
            st.metric(
                "Sector Diversification", 
                f"{div_color} {diversification_score:.0f}/100",
                help="Sector balance: More sectors + balanced allocation + no concentration >25%"
            )
        
        # Show confirmation of data source
        if actual_annual_return_for_synthesis is not None:
            st.success(f"📊 **Annualized return uses actual data from curves**: {actual_annual_return_for_synthesis:.1f}% (matches performance summary above)")
        else:
            st.warning("📈 **Annualized return estimated** - no historical performance data available")
        
        # Sector breakdown
        if len(sector_allocation) > 0:
            max_sector = sector_allocation['weight'].idxmax()
            max_weight = sector_allocation['weight'].max() * 100
            
            st.info(f"""
            **🎯 Sector Diversification Breakdown:**
            • **Sectors**: {num_sectors} sectors → {sector_points:.0f}/50 points (8 points per sector, max 50)
            • **Concentration**: Largest sector {max_sector} ({max_weight:.1f}%) → -{concentration_penalty:.0f} penalty (penalty if >25%)
            • **Balance**: Distribution balance → +{balance_score:.0f}/30 points
            • **Final Score**: {sector_points:.0f} - {concentration_penalty:.0f} + {balance_score:.0f} = {diversification_score:.0f}/100
            """)
        
        # Final optimization insights
        insights = []
        if sharpe_ratio < 0.5:
            insights.append("⚠️ Low Sharpe ratio - consider higher return/lower risk assets")
        if opportunity_margin < -15:
            insights.append("📈 Portfolio may be overvalued vs market")
        if annual_return_display < 8:
            insights.append("📊 Below-market returns - consider growth opportunities")
        if div_yield < 2:
            insights.append("💰 Low dividend yield - consider income-generating stocks")
        if diversification_score < 50:
            insights.append("🎯 Poor diversification - add more sectors")
        
        if insights:
            st.warning("**Final Optimization Opportunities:** " + " • ".join(insights))
        else:
            st.success("✅ **Well-optimized portfolio** - excellent balance across all metrics")
        
        # Export
        st.subheader("📥 Export")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export holdings CSV
            csv_data = display_df.to_csv(index=False)
            st.download_button(
                "📋 Download Holdings CSV",
                data=csv_data,
                file_name=f"portfolio_holdings_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Export synthesis metrics
            synthesis_data = pd.DataFrame([{
                'Metric': 'Sharpe Coefficient',
                'Value': f"{synthesis_metrics['sharpe_ratio']:.2f}"
            }, {
                'Metric': 'Mean Opportunity Margin (%)',
                'Value': f"{synthesis_metrics['mean_opportunity_margin']:+.1f}"
            }, {
                'Metric': 'Annualized Total Return (%)',
                'Value': f"{synthesis_metrics['annualized_total_return']:.1f}"
            }, {
                'Metric': 'Sector Diversification Score',
                'Value': f"{synthesis_metrics['sector_diversification_score']:.0f}"
            }])
            
            synthesis_csv = synthesis_data.to_csv(index=False)
            st.download_button(
                "🎯 Download Synthesis Metrics",
                data=synthesis_csv,
                file_name=f"portfolio_synthesis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        st.success("✅ Analysis complete!")
    
    elif not portfolio_holdings:
        st.info("👈 Upload a CSV file or add stocks manually to begin analysis")
        
        st.markdown("---")
        st.subheader("📊 About Portfolio Optimizer")
        st.markdown("""
        **Synthesis Metrics (5 Total):**
        - **Sharpe Coefficient**: Risk-adjusted return measure (higher = better risk/return)
        - **Mean Opportunity Margin**: Valuation vs market average (+% = undervalued, good!)
        - **Annualized Total Return**: Actual return from historical curves (with dividends reinvested)
        - **Dividend Yield**: Current annual dividend income as % of portfolio value
        - **Sector Diversification**: Portfolio balance score (0-100)
        
        **Sector Diversification Explained:**
        - **Score Components**: Number of sectors (more is better) + Balance (avoid concentration >25%) + Equal distribution
        - **Good Score (70+)**: 6+ sectors, no single sector >25%, balanced allocation
        - **Poor Score (<50)**: Few sectors, high concentration, unbalanced
        - **Example**: Technology 40% + Healthcare 60% = Poor (only 2 sectors, both >25%)
        - **Example**: Tech 20% + Health 20% + Finance 20% + Energy 20% + Utilities 20% = Excellent (5 sectors, balanced)
        
        **CSV Format:** Your CSV should have columns: `symbol`, `shares`
        
        **Adding Stocks:**
        - **Equal Weight**: Automatically balances all holdings (9 stocks + 1 new = 10% each)
        - **Custom Shares**: Specify exact number of shares
        - **Custom Weight**: Target percentage allocation
        """)

if __name__ == "__main__":
    main()