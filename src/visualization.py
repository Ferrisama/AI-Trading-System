import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class DataVisualizer:
    def __init__(self, data):
        self.data = data

    def plot_price_and_anomalies(self, save_path=None):
        """
        Plot the closing price and mark anomalies.
        """
        plt.figure(figsize=(15, 7))
        plt.plot(self.data.index,
                 self.data['Close'], label='Close Price', alpha=0.7)

        # Plot anomalies
        anomalies = self.data[self.data['is_anomaly'] == 1]
        plt.scatter(anomalies.index,
                    anomalies['Close'], color='red', label='Anomaly', zorder=5)

        plt.title('Stock Price with Detected Anomalies')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()

        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_feature_distributions(self, features, save_path=None):
        """
        Plot distributions of selected features, comparing normal vs anomalous points.
        """
        num_features = len(features)
        fig, axes = plt.subplots(num_features, 1, figsize=(
            15, 5*num_features), sharex=False)
        fig.suptitle('Feature Distributions: Normal vs Anomalous', fontsize=16)

        for i, feature in enumerate(features):
            sns.kdeplot(data=self.data, x=feature, hue='is_anomaly', ax=axes[i], palette={0: 'blue', 1: 'red'},
                        shade=True, alpha=0.7)
            axes[i].set_title(f'{feature} Distribution')
            axes[i].legend(['Normal', 'Anomaly'])

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_correlation_heatmap(self, save_path=None):
        """
        Plot a correlation heatmap of the features.
        """
        plt.figure(figsize=(12, 10))
        correlation_matrix = self.data.drop(columns=['is_anomaly']).corr()
        sns.heatmap(correlation_matrix, annot=True,
                    cmap='coolwarm', linewidths=0.5)
        plt.title('Feature Correlation Heatmap')

        if save_path:
            plt.savefig(save_path)
        plt.show()


if __name__ == "__main__":
    # Example usage
    from data_ingestion import DataIngestion
    from preprocessing import DataPreprocessor
    from anomaly_detection import AnomalyDetector
    from datetime import datetime, timedelta

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

    visualizer = DataVisualizer(final_data_with_anomalies)
    visualizer.plot_price_and_anomalies()
    visualizer.plot_feature_distributions(
        ['Returns', 'NormalizedVolume', 'Volatility', 'RSI', 'MACD'])
    visualizer.plot_correlation_heatmap()
