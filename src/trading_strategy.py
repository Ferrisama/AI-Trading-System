import pandas as pd
import numpy as np


class TradingStrategy:
    def __init__(self, data, lookback_period=5, entry_threshold=1, exit_threshold=0.5, max_position=0.1):
        self.data = data
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.max_position = max_position

    def calculate_anomaly_score(self):
        """Calculate a composite anomaly score"""
        columns = ['Returns_anomaly', 'NormalizedVolume_anomaly',
                   'Volatility_anomaly', 'isolation_forest_anomaly']
        self.data['anomaly_score'] = self.data[columns].sum(axis=1)

    def generate_signals(self):
        """Generate trading signals based on anomaly score"""
        self.calculate_anomaly_score()
        signals = pd.DataFrame(index=self.data.index)
        signals['signal'] = 0.0

        # Entry condition
        entry_condition = self.data['anomaly_score'] > self.entry_threshold
        signals.loc[entry_condition, 'signal'] = 1

        # Exit condition
        exit_condition = self.data['anomaly_score'] < self.exit_threshold
        signals.loc[exit_condition, 'signal'] = 0

        # Apply lookback period
        signals['signal'] = signals['signal'].rolling(
            window=self.lookback_period).max()

        # Position sizing
        portfolio_value = 100000  # Assume initial portfolio value of $100,000
        signals['position_size'] = np.minimum(signals['signal'] * (
            self.data['anomaly_score'] / self.data['anomaly_score'].max()) * self.max_position, self.max_position)
        signals['shares'] = (signals['position_size'] *
                             portfolio_value / self.data['Close']).round()

        return signals

    def calculate_returns(self, signals):
        """Calculate returns based on the generated signals"""
        returns = pd.DataFrame(index=signals.index)
        returns['returns'] = self.data['Close'].pct_change()
        returns['strategy_returns'] = signals['shares'].shift(
            1) * returns['returns']
        return returns

    def run_strategy(self):
        """Run the trading strategy and return the results"""
        signals = self.generate_signals()
        returns = self.calculate_returns(signals)
        return pd.concat([signals, returns], axis=1)


if __name__ == "__main__":
    # Example usage
    from data_ingestion import DataIngestion
    from preprocessing import DataPreprocessor
    from anomaly_detection import AnomalyDetector
    from datetime import datetime, timedelta

    # Fetch and preprocess data
    symbol = "AAPL"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    data_ingestion = DataIngestion(symbol, start_date, end_date)
    raw_data = data_ingestion.fetch_data()

    preprocessor = DataPreprocessor(raw_data)
    processed_data = preprocessor.preprocess()
    final_data = preprocessor.add_technical_indicators()

    anomaly_detector = AnomalyDetector(final_data)
    zscore_columns = ['Returns', 'NormalizedVolume', 'Volatility']
    if_columns = ['Returns', 'NormalizedVolume', 'Volatility', 'RSI', 'MACD']
    anomalies = anomaly_detector.combine_anomaly_results(
        zscore_columns, if_columns)

    final_data_with_anomalies = pd.concat([final_data, anomalies], axis=1)

    # Create and run strategy
    strategy = TradingStrategy(final_data_with_anomalies)
    results = strategy.run_strategy()

    print(results.tail())
    print(f"Total Return: {results['strategy_returns'].sum():.2%}")
