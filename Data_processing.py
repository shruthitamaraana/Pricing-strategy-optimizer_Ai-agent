class DataProcessor:
    def __init__(self):
        pass
        
    def clean_price_data(self, df):
        """Clean and format pricing data"""
        # Remove currency symbols and convert to float
        df['price'] = df['price'].replace('[\$,£,€,¥]', '', regex=True).astype(float)
        
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
        df['price_category'] = pd.qcut(df['price'], 4, labels=['budget', 'value', 'premium', 'luxury'])
        
        return df