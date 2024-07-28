import logging
from datetime import datetime, timedelta
import os
from src.portfolio_manager import PortfolioManager
from src.sentiment_analysis import SentimentAnalyzer, incorporate_sentiment
from src.database_operations import DatabaseOperations

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    try:
        symbols = ['AAPL', 'GOOGL', 'MSFT',
                   'AMZN', 'META']  # Changed FB to META
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)  # 2 years of data

        logging.info(f"Starting portfolio analysis for {
                     symbols} from {start_date} to {end_date}")

        # Create a directory for saving plots
        os.makedirs("plots", exist_ok=True)

        # Database configuration
        db_config = {
            "host": "localhost",
            "database": "ai_trading",
            "user": "your_username",
            "password": "your_password",
            "port": "5432"
        }

        # Initialize database operations
        db_ops = DatabaseOperations(db_config)
        db_ops.connect()
        db_ops.create_tables()

        # Initialize and run PortfolioManager
        portfolio_manager = PortfolioManager(symbols, start_date, end_date)

        logging.info("Preparing data for all symbols...")
        portfolio_manager.prepare_data()

        # Store stock data in the database
        for symbol, data in portfolio_manager.data.items():
            db_ops.insert_stock_data(data, symbol)

        logging.info("Optimizing strategies for each symbol...")
        portfolio_manager.optimize_strategies()

        # Log results
        logging.info("\nPortfolio Backtest Results:")
        logging.info(f"Sharpe Ratio: {results['Sharpe Ratio']:.2f}")
        logging.info(f"Total Return: {results['Total Return']:.2%}")
        logging.info(f"Max Drawdown: {results['Max Drawdown']:.2%}")

       # Perform sentiment analysis
        logging.info("Performing sentiment analysis...")
        api_key = "68998fdc9b624e5983bff1f5458cf677"
        sentiment_analyzer = SentimentAnalyzer(api_key)
        sentiment_scores = {}
        for symbol in symbols:
            logging.info(f"Fetching and analyzing sentiment for {symbol}...")
            articles = sentiment_analyzer.fetch_news(symbol)
            logging.info(f"Fetched {len(articles)} articles for {symbol}")
            sentiment = sentiment_analyzer.analyze_sentiment(articles)
            sentiment_scores[symbol] = sentiment
            logging.info(f"Sentiment score for {symbol}: {sentiment}")

        logging.info("Incorporating sentiment into strategies...")
        portfolio_manager = incorporate_sentiment(
            portfolio_manager, sentiment_scores)

        # Run backtests with different allocation strategies
        allocation_methods = ['equal_weight',
                              'risk_parity', 'minimum_variance']
        for method in allocation_methods:
            try:
                logging.info(f"Running portfolio backtest with {
                             method} allocation...")
                results = portfolio_manager.run_portfolio_backtest(
                    allocation_method=method)

                # Log results
                logging.info(
                    f"\nPortfolio Backtest Results ({method} allocation):")
                logging.info(f"Sharpe Ratio: {results['Sharpe Ratio']:.2f}")
                logging.info(f"Total Return: {results['Total Return']:.2%}")
                logging.info(f"Max Drawdown: {results['Max Drawdown']:.2%}")
                logging.info("Allocation:")
                for symbol, weight in results['Allocation'].items():
                    logging.info(f"  {symbol}: {weight:.2%}")

                # Store performance metrics in the database
                db_ops.insert_performance_metrics(
                    results, method, start_date, end_date)

                # Plot and save portfolio performance
                portfolio_manager.plot_portfolio_performance(
                    results['Portfolio Value'],
                    save_path=f"plots/portfolio_performance_{method}.png"
                )
                logging.info(f"Portfolio performance plot for {
                             method} allocation saved in 'plots' directory")
            except Exception as e:
                logging.error(f"An error occurred during {
                              method} allocation: {str(e)}")

        # Individual stock performance
        for symbol in symbols:
            metrics = portfolio_manager.backtesters[symbol].calculate_metrics()
            logging.info(f"\n{symbol} Performance:")
            for key, value in metrics.items():
                logging.info(f"{key}: {value}")

            # Store trading signals in the database
            signals = portfolio_manager.backtesters[symbol].results
            db_ops.insert_trading_signals(signals, symbol, 'combined_strategy')

            portfolio_manager.backtesters[symbol].plot_results(
                save_path=f"plots/{symbol}_performance.png")
            logging.info(
                f"{symbol} performance plot saved in 'plots' directory")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
    finally:
        # Close database connection
        db_ops.disconnect()


if __name__ == "__main__":
    main()
