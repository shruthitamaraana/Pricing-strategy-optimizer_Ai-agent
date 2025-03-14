import plotly.express as px
import plotly.graph_objects as go

class Visualizer:
    def __init__(self):
        pass
        
    def create_competitor_price_comparison(self, df):
        """Create a competitor price comparison chart"""
        fig = px.bar(
            df, 
            x='competitor', 
            y='price',
            color='competitor',
            title='Competitor Price Comparison'
        )
        return fig
        
    def create_price_trend_chart(self, df):
        """Create a price trend chart over time"""
        df_agg = df.groupby(df['timestamp'].dt.to_period('M')).agg({
            'price': 'mean'
        }).reset_index()
        df_agg['timestamp'] = df_agg['timestamp'].dt.to_timestamp()
        
        fig = px.line(
            df_agg,
            x='timestamp',
            y='price',
            title='Price Trends Over Time'
        )
        return fig
        
    def create_pricing_strategy_visualization(self, strategies, competitor_prices):
        """Create a visualization of pricing strategies"""
        fig = go.Figure()
        
        # Add competitor pricing
        fig.add_trace(go.Box(
            y=competitor_prices,
            name='Competitor Prices',
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8
        ))
        
        # Add strategy recommendations
        for strategy, price in strategies.items():
            if strategy != 'price_premium_ratio':
                fig.add_trace(go.Scatter(
                    x=[strategy],
                    y=[price],
                    mode='markers',
                    marker=dict(size=15, color='red'),
                    name=f"{strategy}: ${price:.2f}"
                ))
        
        fig.update_layout(title='Pricing Strategy Recommendations vs. Competitor Pricing')
        return fig