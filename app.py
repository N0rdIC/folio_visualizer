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

    def calculate_sharpe_details(self, metrics: Dict, portfolio_data: pd.DataFrame, historical_performance: pd.DataFrame = None, years: int = 3) -> Dict:
        """Calculate detailed Sharpe ratio breakdown for explanation using actual historical data"""
        risk_free_rate = 0.04  # 4% risk-free rate (10-year Treasury)
        
        # Portfolio expected return components
        dividend_yield = metrics.get('portfolio_dividend_yield', 0) / 100
        
        # Use actual historical price-only return if available, otherwise fall back to estimate
        if historical_performance is not None and not historical_performance.empty and len(historical_performance) > 1:
            # Calculate actual annualized price-only return (capital appreciation without dividends)
            price_only_return_total = historical_performance['Price_Only_Return'].iloc[-1] if 'Price_Only_Return' in historical_performance.columns else 0
            actual_price_growth = ((1 + price_only_return_total/100) ** (1/years) - 1)
            estimated_growth = actual_price_growth
            return_data_source = "actual_historical"
        else:
            # Fall back to market estimate
            estimated_growth = 0.08  # 8% estimated capital appreciation
            return_data_source = "market_estimate"
        
        expected_return = dividend_yield + estimated_growth
        
        # Portfolio risk (volatility) calculation - use actual if available
        portfolio_beta = metrics.get('portfolio_beta', 1.0)
        market_volatility = 0.16  # 16% historical S&P 500 volatility
        
        if historical_performance is not None and not historical_performance.empty and len(historical_performance) > 30:
            # Calculate actual portfolio volatility from historical portfolio values
            # Use Total_Value column which contains actual portfolio value over time
            if 'Total_Value' in historical_performance.columns:
                portfolio_values = historical_performance['Total_Value'].dropna()
                
                if len(portfolio_values) > 10:  # Need sufficient data points
                    # Calculate periodic returns (percentage change between periods)
                    periodic_returns = portfolio_values.pct_change().dropna()
                    
                    if len(periodic_returns) > 5:
                        # Calculate volatility as standard deviation of periodic returns
                        periodic_volatility = periodic_returns.std()
                        # Annualize the volatility (assuming roughly daily data)
                        annual_volatility = periodic_volatility * np.sqrt(252)  # 252 trading days per year
                        portfolio_volatility = annual_volatility
                        volatility_data_source = "actual_historical"
                    else:
                        # Fall back to beta-adjusted market volatility
                        portfolio_volatility = portfolio_beta * market_volatility
                        volatility_data_source = "beta_adjusted"
                else:
                    # Fall back to beta-adjusted market volatility
                    portfolio_volatility = portfolio_beta * market_volatility
                    volatility_data_source = "beta_adjusted"
            else:
                # Fall back to beta-adjusted market volatility
                portfolio_volatility = portfolio_beta * market_volatility
                volatility_data_source = "beta_adjusted"
        else:
            # Fall back to beta-adjusted market volatility
            portfolio_volatility = portfolio_beta * market_volatility
            volatility_data_source = "beta_adjusted"
        
        # Sharpe ratio calculation
        excess_return = expected_return - risk_free_rate
        sharpe_ratio = excess_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'risk_free_rate': risk_free_rate,
            'dividend_yield': dividend_yield,
            'estimated_growth': estimated_growth,
            'expected_return': expected_return,
            'portfolio_beta': portfolio_beta,
            'market_volatility': market_volatility,
            'portfolio_volatility': portfolio_volatility,
            'excess_return': excess_return,
            'return_data_source': return_data_source,
            'volatility_data_source': volatility_data_source,
            'price_growth_annualized_pct': estimated_growth * 100,
            'volatility_annualized_pct': portfolio_volatility * 100
        }

    def identify_problematic_stocks(self, portfolio_df: pd.DataFrame, metrics: Dict, historical_performance: pd.DataFrame = None) -> pd.DataFrame:
        """Identify stocks that may be problematic for portfolio optimization"""
        
        if portfolio_df.empty:
            return pd.DataFrame()
        
        analysis_df = portfolio_df.copy()
        total_value = metrics.get('total_value', 1)
        
        # Calculate weights
        analysis_df['weight'] = (analysis_df['shares'] * analysis_df['current_price']) / total_value
        analysis_df['problem_score'] = 0
        analysis_df['issues'] = ''
        
        # 1. Valuation Issues (P/E Ratio)
        market_pe = 20  # Market average P/E
        for idx, row in analysis_df.iterrows():
            pe_ratio = row['pe_ratio']
            if pe_ratio > 0:
                if pe_ratio > market_pe * 1.5:  # 50% above market
                    analysis_df.loc[idx, 'problem_score'] += 25
                    analysis_df.loc[idx, 'issues'] += f"Overvalued (P/E: {pe_ratio:.1f}); "
                elif pe_ratio > market_pe * 1.2:  # 20% above market
                    analysis_df.loc[idx, 'problem_score'] += 15
                    analysis_df.loc[idx, 'issues'] += f"High P/E ({pe_ratio:.1f}); "
        
        # 2. Risk Issues (Beta)
        for idx, row in analysis_df.iterrows():
            beta = row['beta']
            if beta > 1.5:
                analysis_df.loc[idx, 'problem_score'] += 20
                analysis_df.loc[idx, 'issues'] += f"High Risk (β: {beta:.2f}); "
            elif beta > 1.3:
                analysis_df.loc[idx, 'problem_score'] += 10
                analysis_df.loc[idx, 'issues'] += f"Above-Avg Risk (β: {beta:.2f}); "
        
        # 3. Poor Dividend Performance (for dividend-focused portfolios)
        portfolio_avg_yield = metrics.get('portfolio_dividend_yield', 0)
        if portfolio_avg_yield > 1:  # Only penalize if portfolio focuses on dividends
            for idx, row in analysis_df.iterrows():
                div_yield = row['dividend_yield']
                if div_yield < portfolio_avg_yield * 0.5:  # Less than half portfolio average
                    analysis_df.loc[idx, 'problem_score'] += 15
                    analysis_df.loc[idx, 'issues'] += f"Low Dividend ({div_yield:.1f}%); "
        
        # 4. Concentration Issues
        sector_allocation = analysis_df.groupby('sector')['weight'].sum()
        for idx, row in analysis_df.iterrows():
            sector = row['sector']
            sector_total = sector_allocation.get(sector, 0)
            if sector_total > 0.35:  # Sector >35% of portfolio
                analysis_df.loc[idx, 'problem_score'] += 20
                analysis_df.loc[idx, 'issues'] += f"Over-Concentrated Sector ({sector_total*100:.1f}%); "
            elif sector_total > 0.25:  # Sector >25% of portfolio
                analysis_df.loc[idx, 'problem_score'] += 10
                analysis_df.loc[idx, 'issues'] += f"High Sector Concentration ({sector_total*100:.1f}%); "
        
        # 5. Individual Position Size Issues
        for idx, row in analysis_df.iterrows():
            weight = row['weight']
            if weight > 0.15:  # Single stock >15%
                analysis_df.loc[idx, 'problem_score'] += 25
                analysis_df.loc[idx, 'issues'] += f"Over-Sized Position ({weight*100:.1f}%); "
            elif weight > 0.10:  # Single stock >10%
                analysis_df.loc[idx, 'problem_score'] += 15
                analysis_df.loc[idx, 'issues'] += f"Large Position ({weight*100:.1f}%); "
        
        # 6. Zero Dividend (for income portfolios)
        if portfolio_avg_yield > 2:  # Income-focused portfolio
            for idx, row in analysis_df.iterrows():
                if row['dividend_yield'] == 0:
                    analysis_df.loc[idx, 'problem_score'] += 10
                    analysis_df.loc[idx, 'issues'] += "No Dividend; "
        
        # 7. Market Cap Issues (very small or unknown companies)
        for idx, row in analysis_df.iterrows():
            market_cap = row['market_cap']
            if market_cap > 0 and market_cap < 1e9:  # Less than $1B market cap
                analysis_df.loc[idx, 'problem_score'] += 10
                analysis_df.loc[idx, 'issues'] += f"Small Cap Risk (<$1B); "
        
        # Clean up issues string
        analysis_df['issues'] = analysis_df['issues'].str.rstrip('; ')
        
        # Filter and sort by problem score
        problematic = analysis_df[analysis_df['problem_score'] > 0].copy()
        problematic = problematic.sort_values('problem_score', ascending=False)
        
        # Add severity level
        problematic['severity'] = problematic['problem_score'].apply(
            lambda x: '🔴 Critical' if x >= 50 else '🟠 High' if x >= 30 else '🟡 Medium' if x >= 15 else '🟢 Low'
        )
        
        return problematic.head(10)  # Return top 10 worst

    def calculate_diversification_details(self, metrics: Dict) -> Dict:
        """Calculate detailed diversification score breakdown for explanation"""
        sector_allocation = metrics.get('sector_allocation', pd.DataFrame())
        
        if len(sector_allocation) == 0:
            return {
                'diversification_score': 0,
                'num_sectors': 0,
                'sector_points': 0,
                'max_sector_weight': 0,
                'concentration_penalty': 0,
                'balance_score': 0,
                'sector_breakdown': pd.DataFrame()
            }
        
        num_sectors = len(sector_allocation)
        
        # Component 1: Number of sectors (0-50 points)
        sector_points = min(50, num_sectors * 8)  # 8 points per sector, max 50
        
        # Component 2: Concentration penalty (0-50 penalty)
        max_sector_weight = sector_allocation['weight'].max()
        concentration_penalty = max(0, (max_sector_weight - 0.25) * 100)  # Penalty if >25%
        
        # Component 3: Balance score (0-30 points)
        ideal_weight = 1.0 / num_sectors  # Perfect equal distribution
        deviations = abs(sector_allocation['weight'] - ideal_weight)
        balance_score = max(0, 30 - deviations.sum() * 100)
        
        # Final score (0-100)
        diversification_score = max(0, min(100, sector_points - concentration_penalty + balance_score))
        
        # Create sector breakdown for visualization
        sector_breakdown = sector_allocation.copy()
        sector_breakdown['weight_pct'] = sector_breakdown['weight'] * 100
        sector_breakdown['ideal_weight_pct'] = ideal_weight * 100
        sector_breakdown['deviation'] = abs(sector_breakdown['weight'] - ideal_weight) * 100
        sector_breakdown = sector_breakdown.sort_values('weight', ascending=False)
        
        return {
            'diversification_score': diversification_score,
            'num_sectors': num_sectors,
            'sector_points': sector_points,
            'max_sector_weight': max_sector_weight,
            'concentration_penalty': concentration_penalty,
            'balance_score': balance_score,
            'ideal_weight': ideal_weight,
            'sector_breakdown': sector_breakdown
        }

    def create_sharpe_infographic(self, sharpe_details: Dict):
        """Create visual explanation of Sharpe ratio calculation"""
        
        # Create infographic with plotly
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "📊 Expected Return Components",
                "⚡ Risk (Volatility) Components", 
                "🎯 Sharpe Ratio Formula",
                "📈 Risk-Return Profile"
            ),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"colspan": 2}, None]],
            vertical_spacing=0.20,
            horizontal_spacing=0.1
        )
        
        # Expected Return breakdown
        growth_label = f"Price Growth ({sharpe_details['return_data_source'].replace('_', ' ').title()})"
        return_components = ['Dividend Yield', growth_label, 'Total Expected']
        return_values = [
            sharpe_details['dividend_yield'] * 100,
            sharpe_details['estimated_growth'] * 100,
            sharpe_details['expected_return'] * 100
        ]
        # Use different colors for actual vs estimated data
        colors_return = ['#2E8B57', '#4682B4' if sharpe_details['return_data_source'] == "actual_historical" else '#FFA500', '#228B22']
        
        fig.add_trace(go.Bar(
            x=return_components,
            y=return_values,
            marker_color=colors_return,
            text=[f"{v:.1f}%" for v in return_values],
            textposition='auto',
            name="Expected Return"
        ), row=1, col=1)
        
        # Risk breakdown
        volatility_label = f"Portfolio Vol ({sharpe_details['volatility_data_source'].replace('_', ' ').title()})"
        risk_components = ['Market Vol', 'Portfolio Beta', volatility_label]
        risk_values = [
            sharpe_details['market_volatility'] * 100,
            sharpe_details['portfolio_beta'],
            sharpe_details['portfolio_volatility'] * 100
        ]
        # Use different colors for actual vs estimated volatility
        colors_risk = ['#DC143C', '#FF6347', '#4682B4' if sharpe_details['volatility_data_source'] == "actual_historical" else '#B22222']
        
        fig.add_trace(go.Bar(
            x=risk_components,
            y=risk_values,
            marker_color=colors_risk,
            text=[f"{v:.1f}{'%' if 'Vol' in risk_components[i] else ''}" for i, v in enumerate(risk_values)],
            textposition='auto',
            name="Risk Components"
        ), row=1, col=2)
        
        # Sharpe formula visualization - make it more prominent
        fig.add_trace(go.Scatter(
            x=[0.5, 1.5, 2.5, 3.5],
            y=[2.1, 2.1, 2.1, 2.1],
            mode='text',
            text=[
                f"Expected Return<br><b>{sharpe_details['expected_return']*100:.1f}%</b>",
                f"Risk-Free Rate<br><b>{sharpe_details['risk_free_rate']*100:.1f}%</b>",
                f"Portfolio Volatility<br><b>{sharpe_details['portfolio_volatility']*100:.1f}%</b>",
                f"<b>Sharpe Ratio<br>{sharpe_details['sharpe_ratio']:.2f}</b>"
            ],
            textfont=dict(size=14, color='black'),
            showlegend=False
        ), row=2, col=1)
        
        # Add mathematical symbols
        fig.add_trace(go.Scatter(
            x=[1, 2, 3],
            y=[1.8, 1.8, 1.8],
            mode='text',
            text=["−", "÷", "="],
            textfont=dict(size=24, color='#FF6B35'),
            showlegend=False
        ), row=2, col=1)
        
        # Add formula explanation
        fig.add_trace(go.Scatter(
            x=[2],
            y=[1.3],
            mode='text',
            text=["<b>Sharpe = (Return - Risk-Free) ÷ Risk</b>"],
            textfont=dict(size=16, color='#2E8B57'),
            showlegend=False
        ), row=2, col=1)
        
        fig.update_layout(
            height=600,
            title_text="🎯 Sharpe Ratio Breakdown - Risk-Adjusted Return Measure",
            title_x=0.5,
            showlegend=False,
            # Add background shape for formula section
            shapes=[
                dict(
                    type="rect",
                    xref="x3", yref="y3",
                    x0=0.1, y0=1.1,
                    x1=3.9, y1=2.4,
                    fillcolor="lightblue",
                    opacity=0.15,
                    layer="below",
                    line_width=1,
                    line_color="lightgray"
                )
            ]
        )
        
        fig.update_xaxes(title_text="Components", row=1, col=1)
        fig.update_xaxes(title_text="Risk Factors", row=1, col=2)
        fig.update_yaxes(title_text="Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Value", row=1, col=2)
        
        # Configure formula subplot - make it more visible
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, range=[0, 4], row=2, col=1)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, range=[1, 2.5], row=2, col=1)
        
        return fig

    def create_diversification_infographic(self, div_details: Dict):
        """Create visual explanation of diversification score calculation"""
        
        if div_details['num_sectors'] == 0:
            # Empty portfolio case
            fig = go.Figure()
            fig.add_annotation(
                text="No sectors found in portfolio",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(height=400, title="Sector Diversification Analysis")
            return fig
        
        # Create comprehensive diversification visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "🏗️ Sector Count Score (Max 50 pts)",
                "⚠️ Concentration Penalty", 
                "⚖️ Balance Score (Max 30 pts)",
                "🎯 Final Diversification Score"
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "indicator"}]
            ],
            vertical_spacing=0.15
        )
        
        # 1. Sector count visualization
        sector_count_data = pd.DataFrame({
            'Category': ['Current Sectors', 'Points Earned', 'Max Possible'],
            'Value': [div_details['num_sectors'], div_details['sector_points'], 50],
            'Color': ['#3498db', '#2ecc71', '#95a5a6']
        })
        
        fig.add_trace(go.Bar(
            x=sector_count_data['Category'],
            y=sector_count_data['Value'],
            marker_color=sector_count_data['Color'],
            text=sector_count_data['Value'],
            textposition='auto',
            name="Sector Points"
        ), row=1, col=1)
        
        # 2. Concentration analysis
        max_weight_pct = div_details['max_sector_weight'] * 100
        penalty = div_details['concentration_penalty']
        
        concentration_data = pd.DataFrame({
            'Category': ['Max Sector Weight', 'Threshold (25%)', 'Penalty Applied'],
            'Value': [max_weight_pct, 25, penalty],
            'Color': ['#e74c3c' if max_weight_pct > 25 else '#2ecc71', '#f39c12', '#e74c3c']
        })
        
        fig.add_trace(go.Bar(
            x=concentration_data['Category'],
            y=concentration_data['Value'],
            marker_color=concentration_data['Color'],
            text=[f"{v:.1f}%" if i < 2 else f"{v:.0f} pts" for i, v in enumerate(concentration_data['Value'])],
            textposition='auto',
            name="Concentration"
        ), row=1, col=2)
        
        # 3. Balance score visualization
        balance_data = pd.DataFrame({
            'Category': ['Balance Points', 'Max Possible'],
            'Value': [div_details['balance_score'], 30],
            'Color': ['#9b59b6', '#95a5a6']
        })
        
        fig.add_trace(go.Bar(
            x=balance_data['Category'],
            y=balance_data['Value'],
            marker_color=balance_data['Color'],
            text=balance_data['Value'],
            textposition='auto',
            name="Balance"
        ), row=2, col=1)
        
        # 4. Final score gauge
        score = div_details['diversification_score']
        color = '#2ecc71' if score >= 70 else '#f39c12' if score >= 40 else '#e74c3c'
        
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Final Score"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 40], 'color': "#ffcccc"},
                    {'range': [40, 70], 'color': "#fff3cd"},
                    {'range': [70, 100], 'color': "#d4edda"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ), row=2, col=2)
        
        fig.update_layout(
            height=600,
            title_text="🎯 Sector Diversification Score Breakdown",
            title_x=0.5,
            showlegend=False
        )
        
        return fig

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
        # Check if this is a new file upload by comparing file hash
        uploaded_file_content = uploaded_file.read()
        uploaded_file_hash = hash(uploaded_file_content)
        
        # Only process if this is a new file (different hash) or first time
        if 'last_uploaded_file_hash' not in st.session_state or st.session_state.last_uploaded_file_hash != uploaded_file_hash:
            try:
                csv_content = uploaded_file_content.decode('utf-8')
                imported_holdings = analyzer.import_portfolio_from_csv(csv_content)
                st.session_state.holdings = imported_holdings.copy()
                st.session_state.last_uploaded_file_hash = uploaded_file_hash
                st.sidebar.success(f"✅ Loaded {len(imported_holdings)} holdings")
            except Exception as e:
                st.sidebar.error(f"❌ Error: {str(e)}")
        # If it's the same file, don't reload and overwrite manual additions
    
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
                        shares_to_add = max(1, int(calculated_shares))
                        
                        st.session_state.holdings.append({
                            'symbol': new_symbol, 
                            'shares': shares_to_add,
                            'target_weight': target_weight,
                            'allocation_method': 'equal'
                        })
                        st.sidebar.success(f"Added {new_symbol}: Equal weight ({target_weight*100:.1f}%) - {shares_to_add} shares")
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
    analysis_period = st.sidebar.selectbox("Historical Period", ["1y", "2y", "3y", "5y", "10y"], index=2)
    show_dividend_details = st.sidebar.checkbox("Show Dividend Details", value=False)
    
    # MAIN ANALYSIS
    portfolio_holdings = st.session_state.holdings
    
    if portfolio_holdings and st.sidebar.button("🚀 Analyze Portfolio", type="primary"):
        
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
        
        # Calculate historical performance
        years_str = analysis_period.rstrip('y')  # Remove 'y' suffix
        years = int(years_str) if years_str.isdigit() else 3
        
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
                # Monthly aggregation chart (main chart)
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
                        height=400,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_monthly, use_container_width=True)
                
                # Daily dividends in expander
                with st.expander("📊 View Daily Dividend Payments", expanded=False):
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
                
                # Top dividend payers table in expander
                with st.expander("🏆 View Top Dividend Payers Details", expanded=False):
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
            
            # Recent payments table in expander
            with st.expander("🕒 View Recent Dividend Payments Details", expanded=False):
                display_history = dividend_history.copy()
                display_history['Date'] = display_history['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
                display_history['Dividend_Per_Share'] = display_history['Dividend_Per_Share'].apply(lambda x: f"${x:.4f}")
                display_history['Total_Dividend_Received'] = display_history['Total_Dividend_Received'].apply(lambda x: f"${x:.2f}")
                display_history = display_history.sort_values('Date', ascending=False)
                
                recent_payments = display_history.head(15)
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
        
        # 🎯 SYNTHESIS TABLE (AT THE END WITH ACTUAL DATA AND DETAILED EXPLANATIONS)
        st.header("🎯 Portfolio Synthesis Metrics")
        st.info("📊 **Final synthesis using actual calculated values from analysis above**")
        
        # Calculate detailed breakdowns for both metrics
        sharpe_details = analyzer.calculate_sharpe_details(metrics, portfolio_df, historical_performance, years)
        div_details = analyzer.calculate_diversification_details(metrics)
        
        # Get actual annualized return from historical performance if available
        actual_annual_return_for_synthesis = None
        if not historical_performance.empty and len(historical_performance) > 1:
            total_return_for_synthesis = historical_performance['Total_Return'].iloc[-1]
            actual_annual_return_for_synthesis = ((1 + total_return_for_synthesis/100) ** (1/years) - 1) * 100
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            sharpe_ratio = sharpe_details['sharpe_ratio']
            sharpe_color = "🟢" if sharpe_ratio > 1.0 else "🟡" if sharpe_ratio > 0.5 else "🔴"
            return_icon = "📊" if sharpe_details['return_data_source'] == "actual_historical" else "📈"
            volatility_icon = "📊" if sharpe_details['volatility_data_source'] == "actual_historical" else "⚖️"
            combined_icon = f"{return_icon}{volatility_icon}"
            st.metric(
                "Sharpe Coefficient", 
                f"{sharpe_color} {sharpe_ratio:.2f} {combined_icon}",
                help=f"Risk-adjusted return (>1.0 = good, >2.0 = excellent). Icons: {return_icon}=return data, {volatility_icon}=volatility data"
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
            diversification_score = div_details['diversification_score']
            div_color = "🟢" if diversification_score > 70 else "🟡" if diversification_score > 40 else "🔴"
            st.metric(
                "Sector Diversification", 
                f"{div_color} {diversification_score:.0f}/100",
                help="Sector balance: More sectors + balanced allocation + no concentration >25%"
            )
        
        # DETAILED METRIC EXPLANATIONS WITH INFOGRAPHICS
        st.header("📋 Detailed Metric Explanations")
        
        # Create tabs for detailed explanations
        tab1, tab2 = st.tabs(["🎯 Sharpe Coefficient Breakdown", "🎯 Diversification Score Breakdown"])
        
        with tab1:
            st.subheader("📊 Sharpe Ratio: Risk-Adjusted Return Analysis")
            
            # Create and display Sharpe infographic
            sharpe_fig = analyzer.create_sharpe_infographic(sharpe_details)
            st.plotly_chart(sharpe_fig, use_container_width=True)
            
            # Detailed explanation
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📈 **Return Components**")
                growth_source = "📊 Actual Historical" if sharpe_details['return_data_source'] == "actual_historical" else "📈 Market Estimate"
                st.write(f"• **Dividend Yield**: {sharpe_details['dividend_yield']*100:.1f}% (actual portfolio yield)")
                st.write(f"• **Price Growth**: {sharpe_details['estimated_growth']*100:.1f}% ({growth_source})")
                st.write(f"• **Total Expected Return**: {sharpe_details['expected_return']*100:.1f}%")
                st.write(f"• **Risk-Free Rate**: {sharpe_details['risk_free_rate']*100:.1f}% (10-year Treasury)")
                st.write(f"• **Excess Return**: {sharpe_details['excess_return']*100:.1f}% (premium over risk-free)")
                
                if sharpe_details['return_data_source'] == "actual_historical":
                    st.success(f"✅ **Using actual {years}-year price appreciation from your portfolio!**")
                else:
                    st.info("ℹ️ **Using market estimate** - no historical data available")
            
            with col2:
                st.markdown("### ⚡ **Risk Components**")
                volatility_source = "📊 Actual Portfolio" if sharpe_details['volatility_data_source'] == "actual_historical" else "📈 Beta-Adjusted"
                st.write(f"• **Portfolio Beta**: {sharpe_details['portfolio_beta']:.2f} (vs S&P 500)")
                st.write(f"• **Market Volatility**: {sharpe_details['market_volatility']*100:.0f}% (historical S&P 500)")
                st.write(f"• **Portfolio Volatility**: {sharpe_details['portfolio_volatility']*100:.1f}% ({volatility_source})")
                
                st.markdown("### 🎯 **Final Calculation**")
                st.write(f"**Sharpe = ({sharpe_details['expected_return']*100:.1f}% - {sharpe_details['risk_free_rate']*100:.1f}%) ÷ {sharpe_details['portfolio_volatility']*100:.1f}%**")
                st.write(f"**= {sharpe_details['sharpe_ratio']:.2f}**")
                
                if sharpe_details['volatility_data_source'] == "actual_historical":
                    st.success(f"✅ **Using actual portfolio volatility from {years}-year history!**")
                else:
                    st.info("ℹ️ **Using beta-adjusted volatility** - insufficient historical data")
            
            # Interpretation guide
            st.markdown("---")
            st.markdown("### 🔍 **Sharpe Ratio Interpretation Guide**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**🔴 Poor (< 0.5)**")
                st.write("• Low return for risk taken")
                st.write("• Consider lower-risk alternatives")
                st.write("• May indicate poor stock selection")
            
            with col2:
                st.markdown("**🟡 Acceptable (0.5 - 1.0)**")
                st.write("• Reasonable risk-adjusted return")
                st.write("• Room for improvement")
                st.write("• Monitor and optimize")
            
            with col3:
                st.markdown("**🟢 Excellent (> 1.0)**")
                st.write("• Strong risk-adjusted performance")
                st.write("• Efficient portfolio allocation")
                st.write("• Above-average risk management")
        
        with tab2:
            st.subheader("🎯 Sector Diversification: Balance and Risk Distribution")
            
            # Create and display diversification infographic
            div_fig = analyzer.create_diversification_infographic(div_details)
            st.plotly_chart(div_fig, use_container_width=True)
            
            # Detailed breakdown
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🏗️ **Score Components**")
                st.write(f"• **Number of Sectors**: {div_details['num_sectors']} sectors")
                st.write(f"• **Sector Points**: {div_details['sector_points']:.0f}/50 (8 pts per sector)")
                st.write(f"• **Concentration Penalty**: -{div_details['concentration_penalty']:.0f} pts")
                st.write(f"• **Balance Bonus**: +{div_details['balance_score']:.0f}/30 pts")
                st.write(f"• **Final Score**: {div_details['diversification_score']:.0f}/100")
            
            with col2:
                st.markdown("### ⚖️ **Balance Analysis**")
                if div_details['num_sectors'] > 0:
                    st.write(f"• **Ideal Weight per Sector**: {div_details['ideal_weight']*100:.1f}%")
                    st.write(f"• **Largest Sector Weight**: {div_details['max_sector_weight']*100:.1f}%")
                    
                    if div_details['max_sector_weight'] > 0.25:
                        st.warning(f"⚠️ Concentration risk: Largest sector > 25%")
                    else:
                        st.success(f"✅ Good balance: No sector > 25%")
                else:
                    st.write("• No sector data available")
            
            # Sector breakdown table
            if len(div_details['sector_breakdown']) > 0:
                st.markdown("### 📊 **Sector Weight Analysis**")
                sector_df = div_details['sector_breakdown'].copy()
                sector_df = sector_df.reset_index()
                
                st.dataframe(
                    sector_df[['sector', 'weight_pct', 'ideal_weight_pct', 'deviation']],
                    column_config={
                        "sector": "Sector",
                        "weight_pct": st.column_config.NumberColumn("Current %", format="%.1f%%"),
                        "ideal_weight_pct": st.column_config.NumberColumn("Ideal %", format="%.1f%%"),
                        "deviation": st.column_config.NumberColumn("Deviation", format="%.1f%%")
                    },
                    use_container_width=True,
                    hide_index=True
                )
            
            # Improvement suggestions
            st.markdown("---")
            st.markdown("### 💡 **Diversification Improvement Guide**")
            
            suggestions = []
            if div_details['num_sectors'] < 5:
                suggestions.append("🔹 **Add more sectors** - Target 6+ sectors for better diversification")
            
            if div_details['max_sector_weight'] > 0.3:
                max_sector = div_details['sector_breakdown']['weight_pct'].idxmax() if len(div_details['sector_breakdown']) > 0 else "Unknown"
                suggestions.append(f"🔹 **Reduce concentration** - Largest sector ({max_sector}) > 30%")
            
            if div_details['balance_score'] < 15:
                suggestions.append("🔹 **Rebalance allocation** - Move toward more equal sector weights")
            
            if div_details['diversification_score'] < 50:
                suggestions.append("🔹 **Overall improvement needed** - Consider major portfolio restructuring")
            
            if suggestions:
                for suggestion in suggestions:
                    st.write(suggestion)
            else:
                st.success("✅ **Excellent diversification!** Your portfolio shows strong sector balance.")
        
        # Show confirmation of data sources used
        st.subheader("📊 Data Sources Used")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🎯 Sharpe Ratio Components:**")
            if sharpe_details['return_data_source'] == "actual_historical":
                st.success(f"📊 **Return**: {sharpe_details['estimated_growth']*100:.1f}% actual price growth ({years}y)")
            else:
                st.warning("📈 **Return**: 8.0% market estimate")
                
            if sharpe_details['volatility_data_source'] == "actual_historical":
                st.success(f"📊 **Volatility**: {sharpe_details['portfolio_volatility']*100:.1f}% actual portfolio risk")
            else:
                st.warning(f"⚖️ **Volatility**: {sharpe_details['portfolio_volatility']*100:.1f}% beta-adjusted estimate")
        
        with col2:
            st.markdown("**📈 Total Return:**")
            if actual_annual_return_for_synthesis is not None:
                st.success(f"📊 **Actual**: {actual_annual_return_for_synthesis:.1f}% from historical curves")
            else:
                st.warning("📈 **Estimated**: No historical data available")
        
        with col3:
            st.markdown("**🎯 Accuracy Level:**")
            actual_data_count = sum([
                sharpe_details['return_data_source'] == "actual_historical",
                sharpe_details['volatility_data_source'] == "actual_historical", 
                actual_annual_return_for_synthesis is not None
            ])
            
            if actual_data_count == 3:
                st.success("🟢 **Fully Data-Driven** (3/3 actual)")
            elif actual_data_count == 2:
                st.info("🟡 **Mostly Data-Driven** (2/3 actual)")
            elif actual_data_count == 1:
                st.warning("🟠 **Partially Data-Driven** (1/3 actual)")
            else:
                st.error("🔴 **Estimate-Based** (0/3 actual)")
        
        # 🚨 PORTFOLIO OPTIMIZATION ALERTS
        st.header("🚨 Portfolio Optimization Alerts")
        st.info("🔍 **Automated analysis** - stocks that may need attention for portfolio optimization")
        
        problematic_stocks = analyzer.identify_problematic_stocks(portfolio_df, metrics, historical_performance)
        
        if not problematic_stocks.empty:
            st.subheader("⚠️ Top 5 Stocks to Review")
            
            # Create the problematic stocks table
            display_problematic = problematic_stocks.head(5).copy()
            display_problematic['Weight %'] = (display_problematic['weight'] * 100).round(1)
            display_problematic['Value'] = (display_problematic['shares'] * display_problematic['current_price']).round(2)
            
            # Format for display
            display_cols = ['symbol', 'severity', 'problem_score', 'Weight %', 'Value', 'issues']
            final_display = display_problematic[display_cols].copy()
            final_display.columns = ['Stock', 'Severity', 'Risk Score', 'Weight %', 'Value ($)', 'Issues Identified']
            
            st.dataframe(
                final_display,
                column_config={
                    "Stock": st.column_config.TextColumn("Stock", width="small"),
                    "Severity": st.column_config.TextColumn("Severity", width="small"),
                    "Risk Score": st.column_config.NumberColumn("Risk Score", format="%.0f", width="small"),
                    "Weight %": st.column_config.NumberColumn("Weight %", format="%.1f%%", width="small"),
                    "Value ($)": st.column_config.NumberColumn("Value ($)", format="$%.2f", width="small"),
                    "Issues Identified": st.column_config.TextColumn("Issues Identified", width="large")
                },
                use_container_width=True,
                hide_index=True
            )
            
            # Summary insights
            critical_count = len(problematic_stocks[problematic_stocks['severity'].str.contains('Critical')])
            high_count = len(problematic_stocks[problematic_stocks['severity'].str.contains('High')])
            total_problem_weight = problematic_stocks.head(5)['weight'].sum() * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🔴 Critical Issues", critical_count)
            with col2:
                st.metric("🟠 High Risk Stocks", high_count)
            with col3:
                st.metric("⚠️ Problem Weight", f"{total_problem_weight:.1f}%")
            
            # Action recommendations
            st.markdown("### 💡 **Optimization Recommendations:**")
            
            recommendations = []
            
            if critical_count > 0:
                recommendations.append("🔴 **Immediate Action**: Review critical issues - consider reducing or replacing high-risk positions")
            
            if total_problem_weight > 30:
                recommendations.append("⚖️ **Rebalancing**: Problematic stocks represent significant portfolio weight - consider reallocation")
            
            # Check for specific issues
            overvalued_count = len(problematic_stocks[problematic_stocks['issues'].str.contains('Overvalued|High P/E')])
            concentration_count = len(problematic_stocks[problematic_stocks['issues'].str.contains('Concentrated|Over-Sized')])
            high_risk_count = len(problematic_stocks[problematic_stocks['issues'].str.contains('High Risk')])
            
            if overvalued_count >= 2:
                recommendations.append("📈 **Valuation Check**: Multiple overvalued stocks detected - consider value alternatives")
            
            if concentration_count >= 2:
                recommendations.append("🎯 **Diversification**: High concentration detected - spread risk across more positions")
            
            if high_risk_count >= 2:
                recommendations.append("⚡ **Risk Management**: Multiple high-beta stocks - consider lower-risk alternatives")
            
            if not recommendations:
                recommendations.append("✅ **Good Portfolio Health**: Minor issues detected - consider fine-tuning based on specific concerns")
            
            for rec in recommendations:
                st.write(rec)
                
        else:
            st.success("✅ **Excellent Portfolio Health!** No significant issues detected in your current holdings.")
            st.info("💡 Your portfolio appears well-balanced across valuation, risk, diversification, and position sizing metrics.")
        
        # 📋 COMPREHENSIVE HOLDINGS ANALYSIS
        st.header("📋 Comprehensive Holdings Analysis")
        st.info("🎯 **Individual stock synthesis metrics** - same criteria as Portfolio Synthesis applied to each stock")
        
        # Calculate individual stock historical performance
        individual_stock_data = {}
        
        if not historical_performance.empty and len(portfolio_holdings) > 0:
            st.info("📊 Calculating individual stock historical performance...")
            progress_bar = st.progress(0)
            
            for i, holding in enumerate(portfolio_holdings):
                symbol = holding['symbol']
                progress_bar.progress((i + 1) / len(portfolio_holdings))
                
                try:
                    time.sleep(0.5)  # Rate limiting
                    stock = yf.Ticker(symbol)
                    
                    # Get historical data for the same period
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=years*365)
                    hist = stock.history(start=start_date, end=end_date, auto_adjust=True, back_adjust=True)
                    
                    if not hist.empty and len(hist) > 10:
                        # Calculate total return (price + dividends)
                        initial_price = hist['Close'].iloc[0]
                        final_price = hist['Close'].iloc[-1]
                        
                        # Get dividends over period
                        dividends = stock.dividends
                        if len(dividends) > 0:
                            dividend_dates = dividends.index.tz_localize(None) if dividends.index.tz is not None else dividends.index
                            period_mask = (dividend_dates >= start_date) & (dividend_dates <= end_date)
                            total_dividends_per_share = dividends[period_mask].sum()
                        else:
                            total_dividends_per_share = 0
                        
                        # Calculate total return percentage
                        total_return_pct = ((final_price + total_dividends_per_share) / initial_price - 1) * 100
                        price_only_return_pct = (final_price / initial_price - 1) * 100
                        
                        # Annualize the returns
                        annualized_total_return = ((1 + total_return_pct/100) ** (1/years) - 1) * 100
                        annualized_price_return = ((1 + price_only_return_pct/100) ** (1/years) - 1) * 100
                        
                        # Calculate individual stock volatility
                        returns = hist['Close'].pct_change().dropna()
                        if len(returns) > 5:
                            stock_volatility = returns.std() * np.sqrt(252)  # Annualize
                        else:
                            stock_volatility = None
                        
                        individual_stock_data[symbol] = {
                            'annualized_total_return': annualized_total_return,
                            'annualized_price_return': annualized_price_return,
                            'stock_volatility': stock_volatility,
                            'data_source': 'actual_historical'
                        }
                    
                except Exception as e:
                    # Fallback to estimates
                    individual_stock_data[symbol] = {
                        'annualized_total_return': None,
                        'annualized_price_return': None, 
                        'stock_volatility': None,
                        'data_source': 'estimate'
                    }
            
            progress_bar.empty()
        
        # Create comprehensive holdings dataframe with synthesis metrics
        comprehensive_df = portfolio_df.copy()
        
        # Basic portfolio metrics
        comprehensive_df['Value'] = comprehensive_df['shares'] * comprehensive_df['current_price']
        comprehensive_df['Weight %'] = (comprehensive_df['Value'] / metrics['total_value']) * 100
        
        # Apply individual stock data or fallbacks
        risk_free_rate = 0.04
        market_volatility = 0.16
        
        for idx, row in comprehensive_df.iterrows():
            symbol = row['symbol']
            stock_data = individual_stock_data.get(symbol, {})
            
            # 1. ANNUALIZED TOTAL RETURN per stock (actual or estimated)
            if stock_data.get('annualized_total_return') is not None:
                comprehensive_df.loc[idx, 'Ann_Total_Return'] = stock_data['annualized_total_return']
                comprehensive_df.loc[idx, 'Return_Data_Source'] = 'actual_historical'
            else:
                # Fallback: dividend yield + estimated growth
                estimated_growth = sharpe_details['estimated_growth']*100 if sharpe_details['return_data_source'] == "actual_historical" else 8.0
                comprehensive_df.loc[idx, 'Ann_Total_Return'] = row['dividend_yield'] + estimated_growth
                comprehensive_df.loc[idx, 'Return_Data_Source'] = 'estimate'
            
            # 2. SHARPE COEFFICIENT per stock (using actual volatility if available)
            if stock_data.get('stock_volatility') is not None:
                stock_volatility = stock_data['stock_volatility']
                comprehensive_df.loc[idx, 'Volatility_Data_Source'] = 'actual_historical'
            else:
                stock_volatility = row['beta'] * market_volatility
                comprehensive_df.loc[idx, 'Volatility_Data_Source'] = 'beta_adjusted'
            
            # Expected return for Sharpe (use same as total return calculation)
            if stock_data.get('annualized_total_return') is not None:
                expected_return = stock_data['annualized_total_return'] / 100
            else:
                expected_return = (row['dividend_yield']/100) + (sharpe_details['estimated_growth'] if sharpe_details['return_data_source'] == "actual_historical" else 0.08)
            
            comprehensive_df.loc[idx, 'Stock_Sharpe'] = (expected_return - risk_free_rate) / stock_volatility if stock_volatility > 0 else 0
            
            # 3. MEAN OPPORTUNITY MARGIN per stock (same as before)
            market_pe = 20
            if row['pe_ratio'] > 0:
                comprehensive_df.loc[idx, 'Opp_Margin'] = ((market_pe - row['pe_ratio']) / market_pe) * 100
            else:
                comprehensive_df.loc[idx, 'Opp_Margin'] = 0
        
        # Add color coding/icons for quick assessment
        comprehensive_df['Sharpe_Rating'] = comprehensive_df['Stock_Sharpe'].apply(
            lambda x: "🟢" if x > 1.0 else "🟡" if x > 0.5 else "🔴"
        )
        comprehensive_df['Margin_Rating'] = comprehensive_df['Opp_Margin'].apply(
            lambda x: "🟢" if x > 10 else "🟡" if x > -10 else "🔴"
        )
        comprehensive_df['Return_Rating'] = comprehensive_df['Ann_Total_Return'].apply(
            lambda x: "🟢" if x > 12 else "🟡" if x > 8 else "🔴"
        )
        comprehensive_df['Dividend_Rating'] = comprehensive_df['dividend_yield'].apply(
            lambda x: "🟢" if x > 4 else "🟡" if x > 2 else "🔴"
        )
        
        # Format for display with data source indicators
        display_comprehensive = comprehensive_df.copy()
        display_comprehensive['Sharpe Coef'] = display_comprehensive.apply(
            lambda row: f"{row['Sharpe_Rating']} {row['Stock_Sharpe']:.2f} {'📊' if row.get('Volatility_Data_Source') == 'actual_historical' else '⚖️'}", axis=1
        )
        display_comprehensive['Opp Margin'] = display_comprehensive.apply(
            lambda row: f"{row['Margin_Rating']} {row['Opp_Margin']:+.1f}%", axis=1
        )
        display_comprehensive['Ann Total Return'] = display_comprehensive.apply(
            lambda row: f"{row['Return_Rating']} {row['Ann_Total_Return']:.1f}% {'📊' if row.get('Return_Data_Source') == 'actual_historical' else '📈'}", axis=1
        )
        display_comprehensive['Dividend Yield'] = display_comprehensive.apply(
            lambda row: f"{row['Dividend_Rating']} {row['dividend_yield']:.1f}%", axis=1
        )
        
        # Display the synthesis metrics table
        synthesis_display = display_comprehensive[[
            'symbol', 'name', 'Weight %', 'Value',
            'Sharpe Coef', 'Opp Margin', 'Ann Total Return', 'Dividend Yield'
        ]].copy()
        
        st.dataframe(
            synthesis_display,
            column_config={
                "symbol": st.column_config.TextColumn("Stock", width="small"),
                "name": st.column_config.TextColumn("Company", width="medium"),
                "Weight %": st.column_config.NumberColumn("Weight %", format="%.1f%%", width="small"),
                "Value": st.column_config.NumberColumn("Value", format="$%.0f", width="small"),
                "Sharpe Coef": st.column_config.TextColumn("Sharpe Coefficient", width="medium"),
                "Opp Margin": st.column_config.TextColumn("Opportunity Margin", width="small"), 
                "Ann Total Return": st.column_config.TextColumn("Ann. Total Return", width="medium"),
                "Dividend Yield": st.column_config.TextColumn("Dividend Yield", width="small")
            },
            use_container_width=True,
            hide_index=True
        )
        
        # Data source summary
        actual_return_count = len(comprehensive_df[comprehensive_df.get('Return_Data_Source', '') == 'actual_historical'])
        actual_volatility_count = len(comprehensive_df[comprehensive_df.get('Volatility_Data_Source', '') == 'actual_historical'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Actual Return Data", f"{actual_return_count}/{len(comprehensive_df)}")
        with col2:
            st.metric("📊 Actual Volatility Data", f"{actual_volatility_count}/{len(comprehensive_df)}")
        with col3:
            accuracy_pct = ((actual_return_count + actual_volatility_count) / (len(comprehensive_df) * 2)) * 100
            st.metric("🎯 Data Accuracy", f"{accuracy_pct:.0f}%")
        
        # Synthesis Metrics Legend
        with st.expander("📖 Synthesis Metrics Explained", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **🎯 Sharpe Coefficient (Individual Stock):**
                - Formula: (Expected Return - Risk-Free Rate) ÷ Stock Volatility
                - Expected Return = Actual historical return OR dividend + growth estimate
                - Stock Volatility = Actual historical volatility OR beta-adjusted
                - 📊 = Actual data, ⚖️ = Beta-adjusted estimate
                - 🟢 >1.0 = Excellent, 🟡 0.5-1.0 = Good, 🔴 <0.5 = Poor
                
                **📈 Opportunity Margin (Individual Stock):**
                - Formula: (Market P/E - Stock P/E) ÷ Market P/E × 100%
                - Positive = Undervalued vs market, Negative = Overvalued
                - 🟢 >+10% = Undervalued, 🟡 ±10% = Fair, 🔴 <-10% = Overvalued
                """)
            
            with col2:
                st.markdown("""
                **📊 Annualized Total Return (Individual Stock):**
                - 📊 = Actual historical performance over selected period
                - 📈 = Estimated (dividend yield + growth assumption)
                - Includes dividend reinvestment effect when actual data available
                - 🟢 >12% = Excellent, 🟡 8-12% = Good, 🔴 <8% = Below Market
                
                **💰 Dividend Yield (Individual Stock):**
                - Current annual dividend as % of stock price
                - 🟢 >4% = High Income, 🟡 2-4% = Moderate, 🔴 <2% = Low Income
                
                **Data Source Icons:**
                - 📊 = Actual historical data from selected time period
                - 📈⚖️ = Estimates/calculations (no historical data available)
                """)
        
        # Summary of individual stock synthesis
        excellent_sharpe = len(comprehensive_df[comprehensive_df['Stock_Sharpe'] > 1.0])
        undervalued_stocks = len(comprehensive_df[comprehensive_df['Opp_Margin'] > 10])
        high_return_stocks = len(comprehensive_df[comprehensive_df['Ann_Total_Return'] > 12])
        high_dividend_stocks = len(comprehensive_df[comprehensive_df['dividend_yield'] > 4])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎯 Excellent Sharpe (>1.0)", excellent_sharpe)
        with col2:
            st.metric("📈 Undervalued Stocks", undervalued_stocks)
        with col3:
            st.metric("📊 High Return Stocks (>12%)", high_return_stocks)
        with col4:
            st.metric("💰 High Dividend Stocks (>4%)", high_dividend_stocks)
        
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
            export_df = comprehensive_df[['symbol', 'name', 'sector', 'shares', 'current_price', 'Value', 'Weight %', 'dividend_yield', 'Stock_Sharpe', 'Opp_Margin', 'Ann_Total_Return']].copy()
            csv_data = export_df.to_csv(index=False)
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
                'Value': f"{sharpe_details['sharpe_ratio']:.2f}"
            }, {
                'Metric': 'Mean Opportunity Margin (%)',
                'Value': f"{opportunity_margin:+.1f}"
            }, {
                'Metric': 'Annualized Total Return (%)',
                'Value': f"{annual_return_display:.1f}"
            }, {
                'Metric': 'Sector Diversification Score',
                'Value': f"{diversification_score:.0f}"
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
        
        **Sharpe Coefficient Explained:**
        - **Formula**: (Expected Return - Risk-Free Rate) ÷ Portfolio Volatility
        - **Expected Return**: Portfolio dividend yield + estimated 8% growth
        - **Risk-Free Rate**: 4% (10-year Treasury benchmark)
        - **Portfolio Volatility**: Portfolio beta × 16% market volatility
        - **Good Score (>1.0)**: Excellent risk-adjusted returns
        - **Poor Score (<0.5)**: Low returns for risk taken
        
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