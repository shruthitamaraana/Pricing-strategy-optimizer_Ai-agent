import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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
        """Train the price optimization model"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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