import pandas as pd
import requests
from bs4 import BeautifulSoup
import streamlit as st
import io

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