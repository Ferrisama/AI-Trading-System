import pandas as pd
import numpy as np
from src.portfolio_manager import PortfolioManager
import logging


class WalkForwardOptimization:
    def __init__(self, symbols, start_date, end_date, initial_capital=100000):
        self.symbols = symbols
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.initial_capital = initial_capital

    def optimize(self, train_period=252, test_period=63, step=63, max_iterations=100):
        results = []
        current_start = self.start_date
        iteration = 0

        logging.info(
            f"Starting Walk-Forward Optimization from {self.start_date} to {self.end_date}")
        logging.info(f"Train period: {train_period} days, Test period: {
                     test_period} days, Step: {step} days")

        while current_start + pd.Timedelta(days=train_period+test_period) <= self.end_date and iteration < max_iterations:
            iteration += 1
            train_end = current_start + pd.Timedelta(days=train_period)
            test_end = train_end + pd.Timedelta(days=test_period)

            logging.info(f"Iteration {
                         iteration} - Train: {current_start} to {train_end}, Test: {train_end} to {test_end}")

            try:
                # Train portfolio
                train_portfolio = PortfolioManager(
                    self.symbols, current_start, train_end, self.initial_capital)
                train_portfolio.prepare_data()
                train_portfolio.optimize_strategies()

                # Test portfolio
                test_portfolio = PortfolioManager(
                    self.symbols, train_end, test_end, self.initial_capital)
                test_portfolio.prepare_data()
                test_portfolio.strategies = train_portfolio.strategies  # Use trained strategies

                # Run backtest on test data
                test_results = test_portfolio.run_portfolio_backtest()

                results.append({
                    'Train Start': current_start,
                    'Train End': train_end,
                    'Test Start': train_end,
                    'Test End': test_end,
                    'Sharpe Ratio': test_results['Sharpe Ratio'],
                    'Total Return': test_results['Total Return'],
                    'Max Drawdown': test_results['Max Drawdown']
                })

                logging.info(f"Iteration {iteration} completed successfully. Sharpe Ratio: {
                             test_results['Sharpe Ratio']:.2f}")

            except Exception as e:
                logging.error(f"Error in iteration {iteration}: {
                              str(e)}", exc_info=True)

            current_start += pd.Timedelta(days=step)
            logging.info(f"Moving to next iteration. New start date: {
                         current_start}")

        if iteration == max_iterations:
            logging.warning(f"Reached maximum number of iterations ({
                            max_iterations}). Stopping optimization.")

        if not results:
            logging.warning(
                "No results generated from Walk-Forward Optimization")
        else:
            logging.info(
                f"Walk-Forward Optimization completed with {len(results)} iterations")

        return pd.DataFrame(results)

    def plot_results(self, results, save_path=None):
        """
        Plot the results of walk-forward optimization.

        :param results: DataFrame with optimization results
        :param save_path: Path to save the plot (optional)
        """
        import matplotlib.pyplot as plt

        if results.empty:
            logging.warning("No results to plot")
            return

        plt.figure(figsize=(15, 10))

        plt.subplot(3, 1, 1)
        plt.plot(results['Test End'], results['Sharpe Ratio'])
        plt.title('Walk-Forward Optimization Results')
        plt.ylabel('Sharpe Ratio')

        plt.subplot(3, 1, 2)
        plt.plot(results['Test End'], results['Total Return'])
        plt.ylabel('Total Return')

        plt.subplot(3, 1, 3)
        plt.plot(results['Test End'], results['Max Drawdown'])
        plt.ylabel('Max Drawdown')

        plt.xlabel('Test Period End Date')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logging.info(f"Plot saved to {save_path}")
        else:
            plt.show()
