# Data handling
import yfinance as yf
import pandas as pd
import numpy as np


# Visualization
import matplotlib.pyplot as plt

class Data:
    def __init__(self, ticker):
        self.data = self.fetch_data(ticker)
    
    def fetch_data(self,ticker):
        data = yf.Ticker(ticker=str(ticker))
        hist = data.history(period = "max")
        return hist
    
    def extract_data(self,columns):
        data = self.data[columns]
        return data
    
    def compute_technical_indicators(df):
        # Moving Averages
        df['MA7'] = df['Close'].rolling(window=7).mean()
        df['MA21'] = df['Close'].rolling(window=21).mean()
        
        # Exponential Moving Average
        df['EMA'] = df['Close'].ewm(span=14, adjust=False).mean()
        
        # Bollinger Bands
        df['20SD'] = df['Close'].rolling(window=20).std()
        df['UpperBand'] = df['MA21'] + (df['20SD']*2)
        df['LowerBand'] = df['MA21'] - (df['20SD']*2)
        
        # MACD
        df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff(1)
        delta = delta.dropna()
        up = delta.copy()
        down = delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        time_period = 14
        avg_gain = up.rolling(window=time_period).mean()
        avg_loss = abs(down.rolling(window=time_period).mean())
        RS = avg_gain / avg_loss
        df['RSI'] = 100.0 - (100.0 / (1.0 + RS))
        
        # Return
        df['Return'] = df['Close'].pct_change()
        
        # Drop NaN values
        df = df.dropna()
        
        return df

    

    

