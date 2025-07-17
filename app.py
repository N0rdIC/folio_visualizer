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
        
        # 4. Sector Diversification Score
        sector_allocation = metrics['sector_allocation']
        num_sectors = len(sector_allocation)
        
        # Points for number of sectors
        sector_points = min(50, num_sectors * 8)
        
        # Penalty for concentration
        max_sector_weight = sector_allocation['weight'].max() if len(sector_allocation) > 0 else 0
        concentration_penalty = max(0, (max_sector_weight - 0.25) * 100)
        
        # Balance score
        if num_sectors > 0:
            ideal_weight = 1.0 / num_sectors
            deviations = abs(sector_allocation['weight'] - ideal_weight)
            balance_score = max(0, 30 - deviations.sum() * 100)
        else:
            balance_score = 0
        
        synthesis['sector_diversification_score'] = max(0, min(100, sector_points - concentration_penalty + balance_score))
        
        return synthesis

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
    batch_size = st.sidebar.selectbox("Batch Size", [5, 10, 15], index=1)
    
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
        
        # Calculate metrics
        metrics = analyzer.calculate_portfolio_metrics(portfolio_df)
        synthesis_metrics = analyzer.calculate_synthesis_metrics(portfolio_df, metrics)
        
        # 🎯 SYNTHESIS TABLE (PROMINENT AT TOP)
        st.header("🎯 Portfolio Synthesis Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            sharpe_color = "🟢" if synthesis_metrics['sharpe_ratio'] > 1.0 else "🟡" if synthesis_metrics['sharpe_ratio'] > 0.5 else "🔴"
            st.metric(
                "Sharpe Coefficient", 
                f"{sharpe_color} {synthesis_metrics['sharpe_ratio']:.2f}",
                help="Risk-adjusted return (>1.0 = good, >2.0 = excellent)"
            )
        
        with col2:
            margin_color = "🟢" if synthesis_metrics['mean_opportunity_margin'] > 10 else "🟡" if synthesis_metrics['mean_opportunity_margin'] > -10 else "🔴"
            st.metric(
                "Mean Opp. Margin", 
                f"{margin_color} {synthesis_metrics['mean_opportunity_margin']:+.1f}%",
                help="Valuation vs market (+% = undervalued)"
            )
        
        with col3:
            return_color = "🟢" if synthesis_metrics['annualized_total_return'] > 12 else "🟡" if synthesis_metrics['annualized_total_return'] > 8 else "🔴"
            st.metric(
                "Ann. Total Return (Div)", 
                f"{return_color} {synthesis_metrics['annualized_total_return']:.1f}%",
                help="Expected annual return with dividends"
            )
        
        with col4:
            div_color = "🟢" if synthesis_metrics['sector_diversification_score'] > 70 else "🟡" if synthesis_metrics['sector_diversification_score'] > 40 else "🔴"
            st.metric(
                "Sector Diversification", 
                f"{div_color} {synthesis_metrics['sector_diversification_score']:.0f}/100",
                help="Sector balance score"
            )
        
        # Quick insights
        insights = []
        if synthesis_metrics['sharpe_ratio'] < 0.5:
            insights.append("⚠️ Low Sharpe ratio - consider higher return/lower risk assets")
        if synthesis_metrics['mean_opportunity_margin'] < -15:
            insights.append("📈 Portfolio may be overvalued vs market")
        if synthesis_metrics['annualized_total_return'] < 8:
            insights.append("📊 Below-market returns - consider growth opportunities")
        if synthesis_metrics['sector_diversification_score'] < 50:
            insights.append("🎯 Poor diversification - add more sectors")
        
        if insights:
            st.warning("**Optimization Opportunities:** " + " • ".join(insights))
        else:
            st.success("✅ **Well-optimized portfolio** - good balance across metrics")
        
        st.markdown("---")
        
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
        **Synthesis Metrics:**
        - **Sharpe Coefficient**: Risk-adjusted return measure
        - **Mean Opportunity Margin**: Valuation vs market average
        - **Annualized Total Return**: Expected return with dividends
        - **Sector Diversification**: Portfolio balance score
        
        **CSV Format:** Your CSV should have columns: `symbol`, `shares`
        
        **Adding Stocks:**
        - **Equal Weight**: Automatically balances all holdings
        - **Custom Shares**: Specify exact number of shares
        - **Custom Weight**: Target percentage allocation
        """)

if __name__ == "__main__":
    main()