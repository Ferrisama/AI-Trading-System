import requests
from textblob import TextBlob
import pandas as pd
from datetime import datetime, timedelta


class SentimentAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2/everything"

    def fetch_news(self, symbol, days=7):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        params = {
            'q': symbol,
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
            'language': 'en',
            'apiKey': self.api_key
        }

        response = requests.get(self.base_url, params=params)
        if response.status_code == 200:
            return response.json()['articles']
        else:
            print(f"Error fetching news for {symbol}: {response.status_code}")
            return []

    def analyze_sentiment(self, articles):
        sentiments = []
        for article in articles:
            # Combine title and description, handling cases where either might be None
            text = " ".join(
                filter(None, [article.get('title', ''), article.get('description', '')]))
            if text:
                blob = TextBlob(text)
                sentiments.append(blob.sentiment.polarity)
        return sum(sentiments) / len(sentiments) if sentiments else 0

    def get_sentiment_scores(self, symbols, days=7):
        sentiment_scores = {}
        for symbol in symbols:
            articles = self.fetch_news(symbol, days)
            sentiment_scores[symbol] = self.analyze_sentiment(articles)
        return sentiment_scores


def incorporate_sentiment(portfolio_manager, sentiment_scores, sentiment_weight=0.2):
    for symbol in portfolio_manager.symbols:
        data = portfolio_manager.data[symbol]
        sentiment = sentiment_scores.get(symbol, 0)

        # Scale sentiment to match the range of anomaly scores
        max_anomaly = data['anomaly_score'].max()
        scaled_sentiment = sentiment * (max_anomaly / 2)

        # Combine anomaly score with sentiment
        data['combined_score'] = (
            1 - sentiment_weight) * data['anomaly_score'] + sentiment_weight * scaled_sentiment

        # Update the strategy to use the combined score
        portfolio_manager.strategies[symbol].data = data
        # Override to use combined_score
        portfolio_manager.strategies[symbol].calculate_anomaly_score = lambda: None

    return portfolio_manager
