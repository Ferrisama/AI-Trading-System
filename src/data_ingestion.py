import yfinance as yf
from datetime import datetime, timedelta

class DataIngestion:
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = None

    def fetch_data(self):
        """Fetch historical data from Yahoo Finance."""
        self.data = yf.download(self.symbol, start=self.start_date, end=self.end_date)
        print(f"Fetched {len(self.data)} rows of data for {self.symbol}")
        return self.data

if __name__ == "__main__":
    # Example usage
    symbol = "AAPL"  # Example: Apple Inc.
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # Last 1 year of data

    data_ingestion = DataIngestion(symbol, start_date, end_date)
    data = data_ingestion.fetch_data()
    print(data.head())