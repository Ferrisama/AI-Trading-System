import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


class Backtester:
    def __init__(self, data, strategy, initial_capital=100000):
        self.data = data
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.results = None

    def run(self):
        """
        Run the backtest
        """
        signals = self.strategy.generate_signals()
        self.results = self._calculate_returns(signals)
        return self.results

    def _calculate_returns(self, signals):
        """
        Calculate returns based on the signals
        """
        portfolio = pd.DataFrame(index=signals.index)
        portfolio['holdings'] = signals['signal'] * self.data['Close']
        portfolio['cash'] = self.initial_capital - \
            (signals['signal'].diff().fillna(0) * self.data['Close']).cumsum()
        portfolio['total'] = portfolio['holdings'] + portfolio['cash']
        portfolio['returns'] = portfolio['total'].pct_change()
        return portfolio

    def calculate_metrics(self):
        """
        Calculate performance metrics
        """
        if self.results is None:
            raise ValueError("Backtest hasn't been run yet. Call run() first.")

        total_return = (self.results['total'].iloc[-1] -
                        self.initial_capital) / self.initial_capital
        sharpe_ratio = np.sqrt(
            252) * self.results['returns'].mean() / self.results['returns'].std()

        # Calculate max drawdown
        cumulative_returns = (1 + self.results['returns']).cumprod()
        drawdown = (cumulative_returns.cummax() -
                    cumulative_returns) / cumulative_returns.cummax()
        max_drawdown = drawdown.max()

        # Calculate Sortino ratio
        negative_returns = self.results['returns'][self.results['returns'] < 0]
        sortino_ratio = np.sqrt(
            252) * self.results['returns'].mean() / negative_returns.std()

        # Calculate Calmar ratio
        calmar_ratio = (self.results['returns'].mean() * 252) / max_drawdown

        return {
            'Total Return': total_return,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Sortino Ratio': sortino_ratio,
            'Calmar Ratio': calmar_ratio
        }

    def plot_results(self, save_path=None):
        """
        Plot the results of the backtest
        """
        if self.results is None:
            raise ValueError("Backtest hasn't been run yet. Call run() first.")

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))

        # Plot portfolio value
        ax1.plot(self.results.index, self.results['total'])
        ax1.set_title('Portfolio Value')
        ax1.set_ylabel('Value ($)')

        # Plot returns
        ax2.plot(self.results.index, self.results['returns'])
        ax2.set_title('Daily Returns')
        ax2.set_ylabel('Returns')

        # Plot drawdown
        cumulative_returns = (1 + self.results['returns']).cumprod()
        drawdown = (cumulative_returns.cummax() -
                    cumulative_returns) / cumulative_returns.cummax()
        ax3.plot(self.results.index, drawdown)
        ax3.set_title('Drawdown')
        ax3.set_ylabel('Drawdown')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def monte_carlo_simulation(self, num_simulations=1000, confidence_interval=0.95):
        """
        Perform Monte Carlo simulation
        """
        if self.results is None:
            raise ValueError("Backtest hasn't been run yet. Call run() first.")

        returns = self.results['returns'].dropna()

        simulation_results = []
        for _ in range(num_simulations):
            simulated_returns = np.random.choice(
                returns, size=len(returns), replace=True)
            simulated_value = self.initial_capital * \
                (1 + simulated_returns).cumprod()
            simulation_results.append(simulated_value[-1])

        confidence_level = stats.norm.interval(confidence_interval,
                                               loc=np.mean(simulation_results),
                                               scale=stats.sem(simulation_results))

        return {
            'Mean Final Value': np.mean(simulation_results),
            f'{confidence_interval*100}% Confidence Interval': confidence_level
        }


if __name__ == "__main__":
    # Example usage (you'll need to import your actual strategy and data)
    from trading_strategy import TradingStrategy
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

    # Create strategy and run backtest
    strategy = TradingStrategy(final_data_with_anomalies)
    backtester = Backtester(final_data_with_anomalies, strategy)
    backtester.run()

    # Print metrics
    metrics = backtester.calculate_metrics()
    print("Backtest Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    # Plot results
    backtester.plot_results("backtest_results.png")

    # Run Monte Carlo simulation
    mc_results = backtester.monte_carlo_simulation()
    print("\nMonte Carlo Simulation Results:")
    for key, value in mc_results.items():
        print(f"{key}: {value}")
