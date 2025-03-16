import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import io
import re
import base64

# Set page config
st.set_page_config(
    page_title="Pricing Strategy Optimizer",
    page_icon="💰",
    layout="wide"
)

# Classes implementation
class DataCollector:
    def __init__(self):
        pass
        
    def scrape_competitor_prices(self, url, css_selector):
        """Scrape competitor prices from a given URL using CSS selectors"""
        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(response.content, 'html.parser')
            price_elements = soup.select(css_selector)
            prices = [element.text.strip() for element in price_elements]
            return prices
        except Exception as e:
            return f"Error scraping: {str(e)}"
            
    def fetch_from_api(self, api_url, params=None, headers=None):
        """Fetch pricing data from an API endpoint"""
        try:
            response = requests.get(api_url, params=params, headers=headers)
            return response.json()
        except Exception as e:
            return f"API Error: {str(e)}"
            
    def process_uploaded_file(self, uploaded_file):
        """Process uploaded CSV file with pricing data"""
        try:
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            return f"File processing error: {str(e)}"

class DataProcessor:
    def __init__(self):
        pass
        

    def clean_price_data(self, df):
        """Clean and format pricing data"""
        if 'price' in df.columns:
            # Extract numeric values from the 'price' column
            df['price'] = df['price'].astype(str).apply(lambda x: re.findall(r'\d+\.\d+', x))
            
            # Convert extracted values to float
            df['price'] = df['price'].apply(lambda x: float(x[0]) if x else None)

            # Handle missing values
            df = df.dropna(subset=['price'])

            # Remove outliers (prices that are too high or too low)
            q1 = df['price'].quantile(0.25)
            q3 = df['price'].quantile(0.75)
            iqr = q3 - q1
            df = df[(df['price'] >= q1 - 1.5 * iqr) & (df['price'] <= q3 + 1.5 * iqr)]

        return df

        
    def engineer_features(self, df):
        """Create features for pricing analysis"""
        # Add time-based features if timestamps are available
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
            df['quarter'] = df['timestamp'].dt.quarter
            
        # Calculate price per unit/feature if applicable
        if 'features' in df.columns and 'price' in df.columns:
            df['price_per_feature'] = df['price'] / df['features']
            
        # Calculate price ranges and categories
        if 'price' in df.columns:
            df = df.dropna(subset=['price'])  # Remove rows where price is NaN

            if df['price'].isna().all():
                st.error("Error: Price column contains only NaN values. Cannot categorize.")
                df['price_category'] = None  # Prevent errors
            elif df['price'].nunique() < 4:
                st.warning("Warning: Price column has fewer than 4 unique values. Using simple binning.")
                unique_bins = df['price'].nunique()

                # Use pd.cut instead of pd.qcut when unique values are too few
                df['price_category'] = pd.cut(df['price'], bins=unique_bins, labels=['budget', 'value', 'premium'][:unique_bins])
            else:
                # Add small noise to avoid duplicate bin edges
                df['price'] += np.random.uniform(-0.01, 0.01, df.shape[0])  

                df['price_category'] = pd.qcut(df['price'], 4, labels=['budget', 'value', 'premium', 'luxury'])

        return df



class AnalysisEngine:
    def __init__(self):
        pass
        
    def analyze_competitor_pricing(self, df):
        """Analyze competitor pricing patterns"""
        analysis = {}
        
       # Debugging: Print available columns
        print("DEBUG: Available columns in df:", df.columns)

        # Ensure 'price' column exists
        if 'price' not in df.columns:
            st.error("The 'price' column is missing! Please check your dataset.")
            return {}

        analysis['price_stats'] = {
            'min': df['price'].min(),
            'max': df['price'].max(),
            'mean': df['price'].mean(),
            'median': df['price'].median(),
            'std': df['price'].std()
        }

        
        # Price distribution by competitor
        if 'competitor' in df.columns:
            analysis['competitor_avg_price'] = df.groupby('competitor')['price'].mean().to_dict()
            
        # Price trends over time
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            analysis['price_trends'] = df.groupby(pd.Grouper(key='timestamp', freq='M'))['price'].mean().to_dict()
            
        return analysis
        
    def apply_pricing_psychology(self, price):
        """Apply pricing psychology principles"""
        # Charm pricing (ending in 9)
        charm_price = int(price) - 0.01
        
        # Price anchoring suggestions
        anchor_high = price * 1.5
        anchor_standard = price
        anchor_low = price * 0.8
        
        return {
            'charm_price': charm_price,
            'anchoring_options': {
                'premium': anchor_high,
                'standard': anchor_standard,
                'economy': anchor_low
            }
        }

class PricingModel:
    def __init__(self):
        self.price_model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        self.discount_model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
    def train_price_model(self, X, y):
        # 🔹 Check if X or y is empty before splitting
        if X.empty or y.empty:
            print("Error: Dataset is empty. Please check data preprocessing steps.")
            return None  # Exit function to prevent errors

        # 🔹 Ensure there are enough samples for train-test split
        if len(X) < 2:
            print(f"Error: Not enough samples ({len(X)}) for train-test split. Provide more data.")
            return None
        
        # 🔹 Now safe to split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 🔹 Check if splitting resulted in empty sets
        if X_train.empty or X_test.empty:
            print("Error: Train-test split resulted in an empty set. Try adjusting test_size.")
            return None
        
        # 🔹 Train model
        self.price_model.fit(X_train, y_train)
        score = self.price_model.score(X_test, y_test)
        
        return score

        
    def train_discount_model(self, X, y):
        """Train the discount strategy model"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.discount_model.fit(X_train, y_train)
        score = self.discount_model.score(X_test, y_test)
        return score
        
    def predict_optimal_price(self, features):
        """Predict the optimal price based on features"""
        return self.price_model.predict(features)
        
    def predict_discount_strategy(self, features):
        """Predict the optimal discount strategy"""
        if not hasattr(self.price_model, 'predict'):
            print("Error: Model is not trained yet. Train the model first.")
        else:
            base_price = self.price_model.predict(features)[0]
        return self.discount_model.predict(features)
        
    def recommend_price_strategy(self, features, competitor_prices):
        """Provide a comprehensive pricing strategy"""
        base_price = self.predict_optimal_price(features)[0]
        
        # Adjust based on competitor pricing
        competitor_avg = np.mean(competitor_prices)
        price_premium = base_price / competitor_avg
        
        # Create different strategy options
        strategies = {
            'premium': base_price * 1.1,
            'competitive': base_price,
            'economy': base_price * 0.9,
            'market_average': competitor_avg,
            'price_premium_ratio': price_premium
        }
        
        return strategies

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
        if 'timestamp' not in df.columns:
            return None
            
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df_agg = df.groupby(pd.Grouper(key='timestamp', freq='M')).agg({
            'price': 'mean'
        }).reset_index()
        
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

# Main application

def main():
    # Initialize classes
    data_collector = DataCollector()
    data_processor = DataProcessor()
    analysis_engine = AnalysisEngine()
    pricing_model = PricingModel()
    visualizer = Visualizer()
    
    # Session state initialization
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False

    # Header
    st.title("💹 Pricing Strategy Optimizer")
    st.markdown("""
    Optimize your product pricing with data-driven strategies and AI recommendations.
    This tool analyzes competitor pricing, market trends, and consumer behavior to suggest optimal pricing strategies.-(add files without Null values) 
    """)

    # How It Works - Expandable Section
    with st.expander("ℹ️ How It Works"):
        st.write("""
        1️⃣ **Upload Data** – Provide a CSV file containing product pricing details.  
        2️⃣ **Data Processing** – The system cleans and structures your data.  
        3️⃣ **Analysis & AI Modeling** – AI predicts optimal pricing based on trends.  
        4️⃣ **Visualization & Insights** – Interactive charts and insights help in decision-making.  
        """)

    # User Guide
    st.subheader("📌 Steps to Use:")
    st.markdown("""
    - **Step 1**: Select how you want to input your data from the sidebar.  
    - **Step 2**: Upload a CSV file, scrape competitor prices, or fetch data from an API.  
    - **Step 3**: The app processes the data and generates insights.  
    - **Step 4**: View suggested pricing strategies and visualized insights.  
    """)

    # Display an Image
    st.image(
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTpu_ysYHqqhb4-8Ows_oZPCGSabJVBMrAOog&s",
        caption="Optimize Your Pricing Strategy",
        use_container_width=True
    )

    # Sidebar for input methods
    st.sidebar.title(" 📜 Data Input Methods")
    st.sidebar.markdown("Choose how you want to provide pricing data.")

    input_method = st.sidebar.radio(
        "Select Data Input Method",
        ["Upload CSV", "Web Scraping", "API Integration", "Sample Data"]
    )

    ### 📌 **1. CSV Upload**
    if input_method == "Upload CSV":
        st.sidebar.subheader("📁 Upload Pricing Data")
        st.sidebar.markdown("""
        - Upload a **CSV file** containing product pricing details.  
        - Ensure **no null values** are present in the file.  
        - The system will automatically process and clean your data.  
        """)
        
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            df = data_collector.process_uploaded_file(uploaded_file)
            if isinstance(df, pd.DataFrame):
                st.session_state.data = df
                st.success("✅ Data uploaded successfully!")
            else:
                st.error(df)  # Show error message

    ### 🌐 **2. Web Scraping**
    elif input_method == "Web Scraping":
        st.sidebar.subheader("🌍 Scrape Competitor Prices")
        st.sidebar.markdown("""
        - Enter the **URL** of the website you want to scrape.  
        - Provide the **CSS selector** for price elements.  
        - Enter the **competitor name** to associate with scraped prices.  
        - The system will extract pricing data for analysis.  
        """)
        
        scrape_url = st.sidebar.text_input("🔗 Website URL")
        css_selector = st.sidebar.text_input("🎯 CSS Selector for Price Elements", value=".price")
        competitor_name = st.sidebar.text_input("🏢 Competitor Name")
        
        if st.sidebar.button("🚀 Scrape Prices"):
            if scrape_url and css_selector:
                prices = data_collector.scrape_competitor_prices(scrape_url, css_selector)
                if not isinstance(prices, str):  # If not an error message
                    df = pd.DataFrame({
                        'competitor': [competitor_name] * len(prices),
                        'price': prices,
                        'timestamp': pd.Timestamp.now()
                    })
                    st.session_state.data = df
                    st.success(f"✅ Scraped {len(prices)} prices successfully!")
                else:
                    st.error(prices)  # Show error message
            else:
                st.warning("⚠️ Please enter a valid URL and CSS selector.")

    ### 🔌 **3. API Integration**
    elif input_method == "API Integration":
        st.sidebar.subheader("🔗 Fetch Data from API")
        st.sidebar.markdown("""
        - Enter the **API endpoint URL** that provides pricing data.  
        - If required, enter the **API key** for authentication.  
        - The system will fetch the latest pricing data from the API.  
        """)
        
        api_url = st.sidebar.text_input("🌐 API Endpoint URL")
        api_key = st.sidebar.text_input("🔑 API Key (if required)", type="password")
        
        if st.sidebar.button("📡 Fetch Data"):
            if api_url:
                headers = {'Authorization': f'Bearer {api_key}'} if api_key else None
                api_data = data_collector.fetch_from_api(api_url, headers=headers)
                if not isinstance(api_data, str):  # If not an error message
                    df = pd.DataFrame(api_data)
                    st.session_state.data = df
                    st.success("✅ Data fetched from API successfully!")
                else:
                    st.error(api_data)  # Show error message
            else:
                st.warning("⚠️ Please enter a valid API URL.")

    ### 🎲 **4. Sample Data**
    else:
        st.sidebar.subheader("🎲 Load Sample Data")
        st.sidebar.markdown("""
        - Load pre-generated **sample pricing data**.  
        - Useful for testing features and visualizations.  
        - Random competitor prices will be generated.  
        """)

        if st.sidebar.button("📊 Load Sample Data"):
            competitors = ['CompetitorA', 'CompetitorB', 'CompetitorC', 'CompetitorD']
            products = ['ProductX', 'ProductY', 'ProductZ']
            
            df = pd.DataFrame({
                'competitor': np.random.choice(competitors, 100),
                'product': np.random.choice(products, 100),
                'price': np.random.uniform(9.99, 99.99, 100),
                'features': np.random.randint(1, 10, 100),
                'timestamp': pd.date_range(start='1/1/2023', periods=100)
            })
            
            st.session_state.data = df
            st.success("✅ Sample data loaded successfully!")
    
    # Data Processing
    if st.session_state.data is not None:
        st.subheader("Data Overview")
        st.write(st.session_state.data.head())
        
        st.subheader("Data Processing")
        col1, col2 = st.columns(2)
        with col1:
           if st.button("Clean & Process Data"):
            # Clean and process data
            cleaned_df = data_processor.clean_price_data(st.session_state.data)

            # Debugging: Check if 'price' exists after cleaning
            print("DEBUG: Columns after cleaning:", cleaned_df.columns)

            if 'price' not in cleaned_df.columns:
                st.error("Error: 'price' column is missing after cleaning. Please check the dataset.")
            else:
                processed_df = data_processor.engineer_features(cleaned_df)

                # Debugging: Check if 'price' exists after feature engineering
                print("DEBUG: Columns after feature engineering:", processed_df.columns)

                if 'price' not in processed_df.columns:
                    st.error("Error: 'price' column is missing after feature engineering.")
                else:
                    st.session_state.processed_data = processed_df
                    st.success("Data processed successfully!")

        
        with col2:
            if st.session_state.processed_data is not None:
                if st.button("Run Price Analysis"):
                    # Run analysis
                    analysis_results = analysis_engine.analyze_competitor_pricing(st.session_state.processed_data)
                    st.session_state.analysis_results = analysis_results
                    st.success("Analysis completed!")
    
    # Analysis Results & Visualizations
    if st.session_state.processed_data is not None:
        st.subheader("Processed Data")
        st.write(st.session_state.processed_data.head())
        
        # Price analysis results
        if st.session_state.analysis_results is not None:
            st.subheader("Price Analysis Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Price Statistics:")
                stats = st.session_state.analysis_results['price_stats']
                stats_df = pd.DataFrame({
                    'Metric': list(stats.keys()),
                    'Value': list(stats.values())
                })
                st.write(stats_df)
            
            with col2:
                if 'competitor_avg_price' in st.session_state.analysis_results:
                    st.write("Average Price by Competitor:")
                    comp_prices = st.session_state.analysis_results['competitor_avg_price']
                    comp_df = pd.DataFrame({
                        'Competitor': list(comp_prices.keys()),
                        'Average Price': list(comp_prices.values())
                    })
                    st.write(comp_df)
            
            # Visualizations
            st.subheader("Price Visualizations")
            col1, col2 = st.columns(2)
            
            with col1:
                if 'competitor' in st.session_state.processed_data.columns:
                    fig = visualizer.create_competitor_price_comparison(st.session_state.processed_data)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'timestamp' in st.session_state.processed_data.columns:
                    fig = visualizer.create_price_trend_chart(st.session_state.processed_data)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
    
    # Pricing Model & Optimization
    if st.session_state.processed_data is not None:
        st.subheader("Price Optimization")
        
        # Feature selection for model training
        if not st.session_state.model_trained:
            st.write("Select features for price optimization model:")
            
            numeric_cols = st.session_state.processed_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if 'price' in numeric_cols:
                numeric_cols.remove('price')  # Remove target variable
                
            selected_features = st.multiselect(
                "Select features for model training:",
                options=numeric_cols,
                default=numeric_cols[:min(3, len(numeric_cols))]
            )
            
            if 'price' in st.session_state.processed_data.columns:
                y = st.session_state.processed_data['price']
            else:
                st.error("The 'price' column is missing. Please check the uploaded data or API response.")
                print("DEBUG: Available Columns:", st.session_state.processed_data.columns)
                print("DEBUG: Processed Data Head:\n", st.session_state.processed_data.head())
                y = None  # Prevent crash

            if y is not None and len(selected_features) > 0:
                X = st.session_state.processed_data[selected_features]
                
                # Train model
                score = pricing_model.train_price_model(X, y)
                st.session_state.model_trained = True
                st.session_state.selected_features = selected_features
                st.success(f"Model trained successfully! R² Score: {score:.4f}" if score is not None else "Model trained successfully! R² Score: N/A")
            else:
                st.error("Model training failed. Ensure data contains 'price' and selected features.")


        
        # Price optimization interface
        if st.session_state.model_trained:
            st.subheader("Price Strategy Recommendation")
            print(st.session_state.processed_data.head())  # Print first few rows
            print(st.session_state.processed_data.columns)  # Print column names

            # Input sliders for feature values
        # Ensure processed_data exists and is not empty before using feature
        if 'processed_data' in st.session_state and not st.session_state.processed_data.empty:
            # Initialize feature correctly
            feature = st.selectbox("Select a feature:", st.session_state.processed_data.columns.tolist())

            if feature not in st.session_state.processed_data.columns:
                st.error(f"Error: Column '{feature}' not found in data. Available columns: {st.session_state.processed_data.columns.tolist()}")
            else:
                # Check if the feature is numeric before processing
                if pd.api.types.is_numeric_dtype(st.session_state.processed_data[feature]):
                    min_val = float(st.session_state.processed_data[feature].min())
                else:
                    st.error(f"Feature '{feature}' is not numeric. Please select a numerical feature.")
                    min_val = None  # Avoids errors

            st.write("Adjust product features:")
            feature_values = {}
        else:
            st.error("No processed data found. Please upload or generate data.")
            feature = None  # Ensure feature is initialized

        if "selected_features" not in st.session_state:
            st.session_state.selected_features = []  

            
            for feature in st.session_state.selected_features:
                min_val = float(st.session_state.processed_data[feature].min())
                max_val = float(st.session_state.processed_data[feature].max())
                mean_val = float(st.session_state.processed_data[feature].mean())
                
                # Handle different feature types
                if st.session_state.processed_data[feature].dtype == 'int64':
                    feature_values[feature] = st.slider(
                        f"{feature}:",
                        int(min_val),
                        int(max_val),
                        int(mean_val)
                    )
                else:
                    feature_values[feature] = st.slider(
                        f"{feature}:",
                        float(min_val),
                        float(max_val),
                        float(mean_val)
                    )
            
            # Get competitor prices for comparison
            if 'price' in st.session_state.processed_data.columns:
                competitor_prices = st.session_state.processed_data['price'].values
                
                # Ensure feature_values is defined before use
                if 'feature_values' not in locals():
                    feature_values = {}  # Initialize with an empty dictionary or default values
                
                # Debugging check
                print("Feature Values before DataFrame:", feature_values)
                
                # Prepare features for prediction
                features_df = pd.DataFrame([feature_values])
                
                if st.button("Generate Price Recommendations"):
                    # Get price strategy recommendations
                    strategies = pricing_model.recommend_price_strategy(features_df, competitor_prices)
                    
                    # Apply pricing psychology
                    psychology = analysis_engine.apply_pricing_psychology(strategies['competitive'])
                    
                    # Display recommendations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("### Recommended Pricing Strategies")
                        strategies_df = pd.DataFrame({
                            'Strategy': [k for k in strategies.keys() if k != 'price_premium_ratio'],
                            'Price': [f"${v:.2f}" for k, v in strategies.items() if k != 'price_premium_ratio']
                        })
                        st.write(strategies_df)
                        
                        st.write(f"### Price Premium Ratio: {strategies['price_premium_ratio']:.2f}")
                        st.write("*A ratio > 1 indicates your price is higher than the market average*")

                        
                    with col2:
                        st.write("### Pricing Psychology")
                        st.write(f"Charm Price: ${psychology['charm_price']:.2f}")
                        st.write("Anchoring Options:")
                        for option, price in psychology['anchoring_options'].items():
                            st.write(f"- {option.capitalize()}: ${price:.2f}")
                    
                    # Create visualization
                    st.write("### Price Strategy Visualization")
                    fig = visualizer.create_pricing_strategy_visualization(strategies, competitor_prices)
                    st.plotly_chart(fig, use_container_width=True)
    
    # Export & Download
    if st.session_state.processed_data is not None and st.session_state.analysis_results is not None:
        st.subheader("Export Results")
        
        if st.button("Generate Pricing Report"):
            # Create a report in memory
            buffer = io.BytesIO()
            
            # Create a simple report
            report = f"""# Pricing Strategy Optimization Report
            
## Data Summary
- Total records: {len(st.session_state.processed_data)}
- Date range: {st.session_state.processed_data['timestamp'].min().date() if 'timestamp' in st.session_state.processed_data.columns else 'N/A'} to {st.session_state.processed_data['timestamp'].max().date() if 'timestamp' in st.session_state.processed_data.columns else 'N/A'}

## Price Statistics
- Minimum price: ${st.session_state.analysis_results['price_stats']['min']:.2f}
- Maximum price: ${st.session_state.analysis_results['price_stats']['max']:.2f}
- Average price: ${st.session_state.analysis_results['price_stats']['mean']:.2f}
- Median price: ${st.session_state.analysis_results['price_stats']['median']:.2f}
- Standard deviation: ${st.session_state.analysis_results['price_stats']['std']:.2f}

## Recommendations
- Consider a competitive price point around ${st.session_state.analysis_results['price_stats']['median']:.2f}
- Psychological pricing point: ${int(st.session_state.analysis_results['price_stats']['median']) - 0.01:.2f}
- Premium price point: ${st.session_state.analysis_results['price_stats']['median'] * 1.2:.2f}
- Economy price point: ${st.session_state.analysis_results['price_stats']['median'] * 0.85:.2f}

## Market Positioning
- Setting prices 10-15% above the average may position your product as premium
- Setting prices 5-10% below the average may increase market share
- Matching the median price suggests a market-standard offering
            """
            
            # Write report to buffer
            buffer.write(report.encode())
            buffer.seek(0)
            
            # Create download button
            st.download_button(
                label="Download Pricing Report",
                data=buffer,
                file_name="pricing_strategy_report.md",
                mime="text/markdown"
            )
            
            # Option to download processed data
            csv = st.session_state.processed_data.to_csv(index=False)
            st.download_button(
                label="Download Processed Data",
                data=csv,
                file_name="processed_pricing_data.csv",
                mime="text/csv"
            )

# Run the application
if __name__ == "__main__":
    main()
