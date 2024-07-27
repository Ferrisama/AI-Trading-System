import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class AnomalyDetector:
    def __init__(self, data):
        self.data = data

    def detect_anomalies_zscore(self, columns, threshold=3):
        """
        Detect anomalies using Z-score method.

        Args:
        columns (list): List of column names to check for anomalies
        threshold (float): Number of standard deviations to use as threshold

        Returns:
        DataFrame with anomaly flags
        """
        anomaly_df = pd.DataFrame(index=self.data.index)

        for col in columns:
            col_mean = self.data[col].mean()
            col_std = self.data[col].std()
            z_scores = (self.data[col] - col_mean) / col_std
            anomaly_df[f'{col}_anomaly'] = (
                abs(z_scores) > threshold).astype(int)

        return anomaly_df

    def detect_anomalies_isolation_forest(self, columns, contamination=0.01):
        """
        Detect anomalies using Isolation Forest algorithm.

        Args:
        columns (list): List of column names to use for anomaly detection
        contamination (float): Expected proportion of anomalies in the dataset

        Returns:
        DataFrame with anomaly flags
        """
        clf = IsolationForest(contamination=contamination, random_state=42)
        anomalies = clf.fit_predict(self.data[columns])

        # Convert predictions to binary (0 for inliers, 1 for outliers)
        anomalies = [1 if x == -1 else 0 for x in anomalies]

        return pd.DataFrame(anomalies, index=self.data.index, columns=['isolation_forest_anomaly'])

    def combine_anomaly_results(self, zscore_columns, if_columns, threshold=3, contamination=0.01):
        """
        Combine results from both Z-score and Isolation Forest methods.

        Args:
        zscore_columns (list): Columns to use for Z-score method
        if_columns (list): Columns to use for Isolation Forest method
        threshold (float): Threshold for Z-score method
        contamination (float): Contamination parameter for Isolation Forest

        Returns:
        DataFrame with combined anomaly flags
        """
        zscore_anomalies = self.detect_anomalies_zscore(
            zscore_columns, threshold)
        if_anomalies = self.detect_anomalies_isolation_forest(
            if_columns, contamination)

        combined_anomalies = pd.concat(
            [zscore_anomalies, if_anomalies], axis=1)
        combined_anomalies['is_anomaly'] = (
            combined_anomalies.sum(axis=1) > 0).astype(int)

        return combined_anomalies


if __name__ == "__main__":
    # Example usage
    from data_ingestion import DataIngestion
    from preprocessing import DataPreprocessor
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

    print("Data with anomaly detection:")
    print(final_data_with_anomalies.tail())
    print("\nNumber of detected anomalies:",
          final_data_with_anomalies['is_anomaly'].sum())
