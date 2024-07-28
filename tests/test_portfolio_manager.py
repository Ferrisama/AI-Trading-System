import unittest
from datetime import datetime, timedelta
from src.portfolio_manager import PortfolioManager


class TestPortfolioManager(unittest.TestCase):
    def setUp(self):
        self.symbols = ['AAPL', 'GOOGL', 'MSFT']
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=30)
        self.portfolio_manager = PortfolioManager(
            self.symbols, self.start_date, self.end_date)

    def test_prepare_data(self):
        self.portfolio_manager.prepare_data()
        for symbol in self.symbols:
            self.assertIn(symbol, self.portfolio_manager.data)
            self.assertGreater(len(self.portfolio_manager.data[symbol]), 0)

    def test_optimize_strategies(self):
        self.portfolio_manager.prepare_data()
        self.portfolio_manager.optimize_strategies()
        for symbol in self.symbols:
            self.assertIn(symbol, self.portfolio_manager.strategies)
            self.assertIsNotNone(self.portfolio_manager.strategies[symbol])

    def test_allocate_portfolio(self):
        allocation = self.portfolio_manager.allocate_portfolio()
        self.assertEqual(sum(allocation.values()), 1.0)
        for symbol in self.symbols:
            self.assertIn(symbol, allocation)
            self.assertGreater(allocation[symbol], 0)


if __name__ == '__main__':
    unittest.main()
