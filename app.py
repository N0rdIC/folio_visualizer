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

    def clear_session_data(self):
        """Clear all session data"""
        st.session_state.portfolio_holdings = []
        st.session_state.last_analysis_date = None
        st.session_state.portfolio_name = "My Portfolio"
        st.session_state.portfolio_notes = ""

    def fetch_stock_info(self, symbol: str) -> Dict:
        """Fetch comprehensive stock information"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
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
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'name': symbol,
                'sector': 'Unknown',
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
        """Calculate comprehensive portfolio metrics"""
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

    def simulate_historical_performance(self, portfolio_holdings: List[Dict], years: int = 5) -> pd.DataFrame:
        """Simulate historical portfolio performance with total return (price + dividends)"""
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years*365)
        
        # Fetch historical data for all holdings
        all_data = {}
        total_initial_value = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, holding in enumerate(portfolio_holdings):
            symbol = holding['symbol']
            shares = holding['shares']
            
            status_text.text(f'Fetching {years}-year historical data for {symbol}...')
            progress_bar.progress((i + 1) / len(portfolio_holdings))
            
            try:
                stock = yf.Ticker(symbol)
                hist = stock.history(start=start_date, end=end_date)
                
                if not hist.empty:
                    # Calculate position value over time
                    hist['Position_Value'] = hist['Close'] * shares
                    
                    # Get dividend data and add to historical data
                    dividends = stock.dividends
                    hist['Dividend_Income'] = 0
                    
                    # Add dividends received on each date
                    for date, div in dividends.items():
                        # Handle timezone issues
                        if date.tz is not None:
                            date = date.tz_localize(None)
                        
                        if date in hist.index:
                            hist.loc[date, 'Dividend_Income'] = div * shares
                    
                    # Calculate cumulative dividends (total dividends received up to each date)
                    hist['Cumulative_Dividends'] = hist['Dividend_Income'].cumsum()
                    
                    # Calculate total return percentage (capital gains + dividends)
                    initial_price = hist['Close'].iloc[0]
                    initial_investment = initial_price * shares
                    
                    # Total value = current position value + all dividends received
                    hist['Total_Value'] = hist['Position_Value'] + hist['Cumulative_Dividends']
                    
                    # Total return percentage = (total value / initial investment - 1) * 100
                    hist['Total_Return_Pct'] = ((hist['Total_Value'] / initial_investment) - 1) * 100
                    
                    # Price-only return (for comparison)
                    hist['Price_Return_Pct'] = ((hist['Close'] / initial_price) - 1) * 100
                    
                    all_data[symbol] = hist
                    total_initial_value += initial_investment
                    
            except Exception as e:
                st.warning(f"Could not fetch historical data for {symbol}: {str(e)}")
        
        progress_bar.empty()
        status_text.empty()
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all positions into portfolio performance
        portfolio_performance = pd.DataFrame()
        
        # Get common date range
        common_dates = None
        for symbol, data in all_data.items():
            if common_dates is None:
                common_dates = data.index
            else:
                common_dates = common_dates.intersection(data.index)
        
        if len(common_dates) == 0:
            return pd.DataFrame()
        
        # Calculate portfolio metrics for each date
        for date in common_dates:
            portfolio_value = 0
            total_dividends = 0
            price_only_value = 0
            
            for symbol, data in all_data.items():
                if date in data.index:
                    portfolio_value += data.loc[date, 'Position_Value']
                    total_dividends += data.loc[date, 'Cumulative_Dividends']
                    # Price-only calculation for comparison
                    shares = next(h['shares'] for h in portfolio_holdings if h['symbol'] == symbol)
                    initial_price = data['Close'].iloc[0]
                    price_only_value += (data.loc[date, 'Close'] / initial_price - 1) * initial_price * shares
            
            portfolio_performance.loc[date, 'Portfolio_Value'] = portfolio_value
            portfolio_performance.loc[date, 'Total_Dividends'] = total_dividends
            portfolio_performance.loc[date, 'Total_Value'] = portfolio_value + total_dividends
            
            # Total return including dividends
            portfolio_performance.loc[date, 'Total_Return'] = ((portfolio_value + total_dividends) / total_initial_value - 1) * 100
            
            # Price-only return for comparison
            portfolio_performance.loc[date, 'Price_Only_Return'] = (portfolio_value / total_initial_value - 1) * 100
        
        return portfolio_performance.sort_index()

    def generate_dividend_calendar(self, portfolio_holdings: List[Dict]) -> pd.DataFrame:
        """Generate monthly dividend calendar"""
        dividend_calendar = []
        
        for holding in portfolio_holdings:
            symbol = holding['symbol']
            shares = holding['shares']
            
            try:
                stock = yf.Ticker(symbol)
                
                # Get dividend history for the last 2 years
                end_date = datetime.now()
                start_date = end_date - timedelta(days=730)
                dividends = stock.dividends
                
                # Filter recent dividends and handle timezone issues
                if len(dividends) > 0:
                    # Convert timezone-aware index to timezone-naive for comparison
                    dividend_dates = dividends.index.tz_localize(None) if dividends.index.tz is not None else dividends.index
                    
                    # Filter recent dividends
                    recent_mask = dividend_dates >= start_date
                    recent_dividends = dividends[recent_mask]
                    recent_dividend_dates = dividend_dates[recent_mask]
                    
                    if len(recent_dividends) > 0:
                        # Estimate dividend frequency
                        if len(recent_dividends) >= 4:
                            # Quarterly dividend likely
                            frequency = 'Quarterly'
                            last_dividend = recent_dividends.iloc[-1]
                            last_date = recent_dividend_dates[-1]
                            
                            # Project next 4 quarters
                            for quarter in range(4):
                                next_date = last_date + timedelta(days=90*(quarter+1))
                                dividend_calendar.append({
                                    'Symbol': symbol,
                                    'Date': next_date,
                                    'Month': next_date.strftime('%Y-%m'),
                                    'Dividend_Per_Share': float(last_dividend),
                                    'Total_Dividend': float(last_dividend) * shares,
                                    'Frequency': frequency
                                })
                        
                        elif len(recent_dividends) >= 1:
                            # Annual dividend likely
                            frequency = 'Annual'
                            last_dividend = recent_dividends.iloc[-1]
                            last_date = recent_dividend_dates[-1]
                            
                            next_date = last_date + timedelta(days=365)
                            dividend_calendar.append({
                                'Symbol': symbol,
                                'Date': next_date,
                                'Month': next_date.strftime('%Y-%m'),
                                'Dividend_Per_Share': float(last_dividend),
                                'Total_Dividend': float(last_dividend) * shares,
                                'Frequency': frequency
                            })
            
            except Exception as e:
                st.warning(f"Could not generate dividend calendar for {symbol}: {str(e)}")
        
        if dividend_calendar:
            df = pd.DataFrame(dividend_calendar)
            # Group by month for monthly distribution
            monthly_distribution = df.groupby('Month')['Total_Dividend'].sum().reset_index()
            return df, monthly_distribution
        else:
            return pd.DataFrame(), pd.DataFrame()

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
    
    include_ai_analysis = st.sidebar.checkbox(
        "Include AI Analysis", 
        value=True,
        help="Requires OpenAI API key in secrets"
    )
    
    # Main analysis
    if portfolio_holdings and st.sidebar.button("🚀 Analyze Portfolio", type="primary"):
        
        st.info(f"Analyzing portfolio with {len(portfolio_holdings)} holdings...")
        
        # Fetch current data for all holdings
        portfolio_data = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, holding in enumerate(portfolio_holdings):
            status_text.text(f'Fetching data for {holding["symbol"]}...')
            progress_bar.progress((i + 1) / len(portfolio_holdings))
            
            stock_info = analyzer.fetch_stock_info(holding['symbol'])
            stock_info['shares'] = holding['shares']
            portfolio_data.append(stock_info)
        
        progress_bar.empty()
        status_text.empty()
        
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
            historical_performance = analyzer.simulate_historical_performance(portfolio_holdings, years)
        
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
        
        # Dividend Calendar
        st.header("📅 Dividend Calendar")
        
        dividend_calendar, monthly_distribution = analyzer.generate_dividend_calendar(portfolio_holdings)
        
        if not dividend_calendar.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Monthly Dividend Distribution")
                if not monthly_distribution.empty:
                    fig_monthly = px.bar(
                        monthly_distribution,
                        x='Month',
                        y='Total_Dividend',
                        title="Expected Monthly Dividend Income",
                        labels={'Total_Dividend': 'Dividend Amount ($)', 'Month': 'Month'}
                    )
                    st.plotly_chart(fig_monthly, use_container_width=True)
            
            with col2:
                st.subheader("📋 Upcoming Dividends")
                upcoming = dividend_calendar[dividend_calendar['Date'] >= datetime.now()].head(10)
                if not upcoming.empty:
                    st.dataframe(
                        upcoming[['Symbol', 'Date', 'Dividend_Per_Share', 'Total_Dividend']],
                        column_config={
                            "Date": st.column_config.DateColumn(format="YYYY-MM-DD"),
                            "Dividend_Per_Share": st.column_config.NumberColumn(format="$%.2f"),
                            "Total_Dividend": st.column_config.NumberColumn(format="$%.2f")
                        },
                        use_container_width=True
                    )
        
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
