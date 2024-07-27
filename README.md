# AI Trading System

This project implements an AI-driven trading system that uses anomaly detection, sentiment analysis, and various portfolio allocation strategies to make trading decisions.

## Project Structure

```
ai_trading_project/
│
├── src/
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   ├── anomaly_detection.py
│   ├── trading_strategy.py
│   ├── backtesting.py
│   ├── portfolio_manager.py
│   └── sentiment_analysis.py
│
├── main.py
├── requirements.txt
└── README.md
```

## Features

- Data ingestion from Yahoo Finance
- Data preprocessing and feature engineering
- Anomaly detection using Z-score and Isolation Forest methods
- Sentiment analysis of news articles
- Multiple portfolio allocation strategies (Equal Weight, Risk Parity, Minimum Variance)
- Comprehensive backtesting framework

## Setup

1. Clone the repository:

   ```
   git clone https://github.com/Ferrisama/AI-Trading-System.git
   cd AI-Trading-System
   ```

2. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

3. Sign up for a free API key at [NewsAPI](https://newsapi.org/)

4. Replace `"YOUR_NEWS_API_KEY"` in `main.py` with your actual API key.

## Usage

Run the main script:

```
python main.py
```

This will perform the following steps:

1. Fetch and preprocess data for the specified stocks
2. Detect anomalies in the data
3. Perform sentiment analysis on recent news
4. Run backtests with different portfolio allocation strategies
5. Generate performance metrics and plots

## Results

The results, including performance metrics and plots, will be saved in the `plots/` directory.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
.
