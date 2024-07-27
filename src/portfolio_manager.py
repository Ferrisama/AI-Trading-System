import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as GridSpec
import seaborn as sns
import logging
from scipy.optimize import minimize
from src.data_ingestion import DataIngestion
from src.preprocessing import DataPreprocessor
from src.anomaly_detection import AnomalyDetector
from src.trading_strategy import TradingStrategy
from src.backtesting import Backtester


class PortfolioManager:
    def __init__(self, symbols, start_date, end_date, initial_capital=100000):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.data = {}
        self.strategies = {}
        self.backtesters = {}

    def prepare_data(self):
        for symbol in self.symbols:
            try:
                data_ingestion = DataIngestion(
                    symbol, self.start_date, self.end_date)
                raw_data = data_ingestion.fetch_data()

                if raw_data.empty:
                    logging.warning(f"No data available for {
                                    symbol}. Skipping this symbol.")
                    continue

                preprocessor = DataPreprocessor(raw_data)
                processed_data = preprocessor.preprocess()
                final_data = preprocessor.add_technical_indicators()

                # Ensure all required columns are present
                required_columns = [
                    'Returns', 'NormalizedVolume', 'Volatility', 'RSI', 'MACD']
                missing_columns = [
                    col for col in required_columns if col not in final_data.columns]
                if missing_columns:
                    logging.warning(f"Missing columns for {symbol}: {
                                    missing_columns}. Skipping anomaly detection for these columns.")
                    for col in missing_columns:
                        final_data[col] = 0  # Add dummy column

                anomaly_detector = AnomalyDetector(final_data)
                zscore_columns = [col for col in [
                    'Returns', 'NormalizedVolume', 'Volatility'] if col in final_data.columns]
                if_columns = [col for col in ['Returns', 'NormalizedVolume',
                                              'Volatility', 'RSI', 'MACD'] if col in final_data.columns]
                anomalies = anomaly_detector.combine_anomaly_results(
                    zscore_columns, if_columns)

                self.data[symbol] = pd.concat([final_data, anomalies], axis=1)
                logging.info(f"Successfully prepared data for {symbol}")
            except Exception as e:
                logging.error(f"Error preparing data for {symbol}: {str(e)}")

    def optimize_strategies(self):
        for symbol, data in self.data.items():
            try:
                best_sharpe = -np.inf
                best_params = {}

                for lookback_period in [3, 5, 7]:
                    for entry_threshold in [0.5, 1, 1.5]:
                        for exit_threshold in [0.3, 0.5, 0.7]:
                            for max_position in [0.05, 0.1, 0.15]:
                                strategy = TradingStrategy(
                                    data, lookback_period, entry_threshold, exit_threshold, max_position)
                                backtester = Backtester(data, strategy)
                                backtester.run()
                                metrics = backtester.calculate_metrics()

                                if metrics['Sharpe Ratio'] > best_sharpe:
                                    best_sharpe = metrics['Sharpe Ratio']
                                    best_params = {
                                        'lookback_period': lookback_period,
                                        'entry_threshold': entry_threshold,
                                        'exit_threshold': exit_threshold,
                                        'max_position': max_position
                                    }

                self.strategies[symbol] = TradingStrategy(data, **best_params)
                self.backtesters[symbol] = Backtester(
                    data, self.strategies[symbol])
                self.backtesters[symbol].run()
                logging.info(f"Successfully optimized strategy for {symbol}")
            except Exception as e:
                logging.error(f"Error optimizing strategy for {
                              symbol}: {str(e)}")

    def allocate_portfolio(self, method='equal_weight'):
        if method == 'equal_weight':
            return self._equal_weight_allocation()
        elif method == 'risk_parity':
            return self._risk_parity_allocation()
        elif method == 'minimum_variance':
            return self._minimum_variance_allocation()
        else:
            raise ValueError("Unknown allocation method")

    def _equal_weight_allocation(self):
        return {symbol: 1/len(self.symbols) for symbol in self.symbols}

    def _risk_parity_allocation(self):
        returns = pd.DataFrame(
            {symbol: self.backtesters[symbol].results['returns'] for symbol in self.symbols})
        cov_matrix = returns.cov()

        def risk_budget_objective(weights, args):
            cov_matrix = args[0]
            portfolio_risk = np.sqrt(
                np.dot(np.dot(weights, cov_matrix), weights.T))
            risk_contribution = np.dot(cov_matrix, weights.T) / portfolio_risk
            risk_target = portfolio_risk / len(weights)
            return np.sum((risk_contribution - risk_target)**2)

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(len(self.symbols)))
        initial_weights = [1/len(self.symbols)] * len(self.symbols)

        result = minimize(risk_budget_objective, initial_weights, args=[
                          cov_matrix], method='SLSQP', constraints=constraints, bounds=bounds)

        return {symbol: weight for symbol, weight in zip(self.symbols, result.x)}

    def _minimum_variance_allocation(self):
        returns = pd.DataFrame(
            {symbol: self.backtesters[symbol].results['returns'] for symbol in self.symbols})
        cov_matrix = returns.cov()

        def portfolio_variance(weights, cov_matrix):
            return np.dot(np.dot(weights, cov_matrix), weights.T)

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(len(self.symbols)))
        initial_weights = [1/len(self.symbols)] * len(self.symbols)

        result = minimize(portfolio_variance, initial_weights, args=(
            cov_matrix,), method='SLSQP', constraints=constraints, bounds=bounds)

        return {symbol: weight for symbol, weight in zip(self.symbols, result.x)}

    def run_portfolio_backtest(self, allocation_method='equal_weight'):
        allocation = self.allocate_portfolio(method=allocation_method)
        portfolio_value = pd.DataFrame(
            index=self.data[self.symbols[0]].index, columns=['Total'])
        portfolio_value['Total'] = self.initial_capital

        for symbol in self.symbols:
            strategy_returns = self.backtesters[symbol].results['returns']
            portfolio_value[symbol] = self.initial_capital * \
                allocation[symbol] * (1 + strategy_returns).cumprod()

        portfolio_value['Total'] = portfolio_value.sum(axis=1)
        portfolio_returns = portfolio_value['Total'].pct_change()

        sharpe_ratio = np.sqrt(
            252) * portfolio_returns.mean() / portfolio_returns.std()
        total_return = (
            portfolio_value['Total'].iloc[-1] - self.initial_capital) / self.initial_capital
        max_drawdown = (portfolio_value['Total'] /
                        portfolio_value['Total'].cummax() - 1).min()

        return {
            'Portfolio Value': portfolio_value,
            'Sharpe Ratio': sharpe_ratio,
            'Total Return': total_return,
            'Max Drawdown': max_drawdown,
            'Allocation': allocation
        }

    def plot_comprehensive_results(self, save_path=None):
        """
        Create a comprehensive plot of portfolio and individual stock performances.
        """
        # Set up the plot
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(3, 2, figure=fig)

        # Plot 1: Portfolio Value
        ax1 = fig.add_subplot(gs[0, :])
        for column in self.results['Portfolio Value'].columns:
            ax1.plot(self.results['Portfolio Value'].index,
                     self.results['Portfolio Value'][column], label=column)
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Value ($)')
        ax1.legend()
        ax1.grid(True)

        # Plot 2: Portfolio Returns Distribution
        ax2 = fig.add_subplot(gs[1, 0])
        portfolio_returns = self.results['Portfolio Value']['Total'].pct_change(
        )
        sns.histplot(portfolio_returns, kde=True, ax=ax2)
        ax2.set_title('Distribution of Portfolio Returns')
        ax2.set_xlabel('Daily Returns')
        ax2.set_ylabel('Frequency')

        # Plot 3: Drawdown
        ax3 = fig.add_subplot(gs[1, 1])
        portfolio_value = self.results['Portfolio Value']['Total']
        drawdown = (portfolio_value / portfolio_value.cummax() - 1)
        ax3.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax3.set_title('Portfolio Drawdown')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Drawdown')
        ax3.grid(True)

        # Plot 4: Individual Stock Performances
        ax4 = fig.add_subplot(gs[2, :])
        for symbol in self.strategies.keys():
            returns = (
                1 + self.backtesters[symbol].results['strategy_returns']).cumprod()
            ax4.plot(returns.index, returns, label=symbol)
        ax4.set_title('Individual Stock Strategy Performances')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Cumulative Returns')
        ax4.legend()
        ax4.grid(True)

        # Format dates on x-axis
        for ax in [ax1, ax3, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()

    def plot_portfolio_performance(self, portfolio_value, save_path=None):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(15, 10))
        for column in portfolio_value.columns:
            plt.plot(portfolio_value.index,
                     portfolio_value[column], label=column)

        plt.title('Portfolio Performance')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def generate_performance_report(self, save_path=None):
        """
        Generate a comprehensive performance report.
        """
        report = "Portfolio Performance Report\n"
        report += "===========================\n\n"

        report += "Overall Portfolio Performance:\n"
        report += f"Total Return: {self.results['Total Return']:.2%}\n"
        report += f"Sharpe Ratio: {self.results['Sharpe Ratio']:.2f}\n"
        report += f"Max Drawdown: {self.results['Max Drawdown']:.2%}\n\n"

        report += "Individual Stock Performances:\n"
        for symbol in self.strategies.keys():
            metrics = self.backtesters[symbol].calculate_metrics()
            report += f"{symbol}:\n"
            report += f"  Total Return: {metrics['Total Return']:.2%}\n"
            report += f"  Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}\n"
            report += f"  Max Drawdown: {metrics['Max Drawdown']:.2%}\n"
            report += f"  Sortino Ratio: {metrics['Sortino Ratio']:.2f}\n"
            report += f"  Calmar Ratio: {metrics['Calmar Ratio']:.2f}\n\n"

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        else:
            print(report)

        return report


if __name__ == "__main__":
    # Example usage remains the same
    pass
