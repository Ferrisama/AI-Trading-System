import pandas as pd
import numpy as np


class DataPreprocessor:
    def __init__(self, data):
        self.data = data

    def preprocess(self):
        """Preprocess the data: calculate returns, normalize volume, handle missing values."""
        if self.data is None or self.data.empty:
            print("No data to preprocess. Please fetch data first.")
            return None

        # Calculate returns
        self.data['Returns'] = self.data['Close'].pct_change()

        # Normalize volume
        self.data['NormalizedVolume'] = (
            self.data['Volume'] - self.data['Volume'].mean()) / self.data['Volume'].std()

        # Calculate price change
        self.data['PriceChange'] = self.data['Close'] - self.data['Open']

        # Calculate moving averages
        self.data['MA5'] = self.data['Close'].rolling(window=5).mean()
        self.data['MA20'] = self.data['Close'].rolling(window=20).mean()

        # Calculate volatility (using 20-day rolling standard deviation of returns)
        self.data['Volatility'] = self.data['Returns'].rolling(window=20).std()

        # Remove any rows with NaN values
        self.data.dropna(inplace=True)

        print("Data preprocessing completed.")
        return self.data

    def add_technical_indicators(self):
        """Add some basic technical indicators."""
        # Relative Strength Index (RSI)
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))

        # Moving Average Convergence Divergence (MACD)
        exp1 = self.data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = self.data['Close'].ewm(span=26, adjust=False).mean()
        self.data['MACD'] = exp1 - exp2
        self.data['Signal Line'] = self.data['MACD'].ewm(
            span=9, adjust=False).mean()

        # Remove any new NaN values
        self.data.dropna(inplace=True)

        print("Technical indicators added.")
        return self.data


if __name__ == "__main__":
    # Example usage
    from data_ingestion import DataIngestion
    from datetime import datetime, timedelta

    symbol = "AAPL"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    data_ingestion = DataIngestion(symbol, start_date, end_date)
    raw_data = data_ingestion.fetch_data()

    preprocessor = DataPreprocessor(raw_data)
    processed_data = preprocessor.preprocess()
    final_data = preprocessor.add_technical_indicators()

    print(final_data.tail())
    print(final_data.columns)
