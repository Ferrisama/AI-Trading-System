import psycopg2
from psycopg2 import sql
import pandas as pd
from sqlalchemy import create_engine


class DatabaseOperations:
    def __init__(self, db_config):
        self.db_config = db_config
        self.conn = None
        self.cursor = None

    def connect(self):
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.cursor = self.conn.cursor()
            print("Connected to the database successfully.")
        except (Exception, psycopg2.Error) as error:
            print("Error while connecting to PostgreSQL", error)

    def disconnect(self):
        if self.conn:
            self.cursor.close()
            self.conn.close()
            print("PostgreSQL connection is closed")

    def create_tables(self):
        commands = (
            """
            CREATE TABLE IF NOT EXISTS stock_data (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(10) NOT NULL,
                date DATE NOT NULL,
                open FLOAT NOT NULL,
                high FLOAT NOT NULL,
                low FLOAT NOT NULL,
                close FLOAT NOT NULL,
                volume BIGINT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS trading_signals (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(10) NOT NULL,
                date DATE NOT NULL,
                signal FLOAT NOT NULL,
                strategy VARCHAR(50) NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id SERIAL PRIMARY KEY,
                strategy VARCHAR(50) NOT NULL,
                start_date DATE NOT NULL,
                end_date DATE NOT NULL,
                total_return FLOAT NOT NULL,
                sharpe_ratio FLOAT NOT NULL,
                max_drawdown FLOAT NOT NULL
            )
            """
        )
        try:
            for command in commands:
                self.cursor.execute(command)
            self.conn.commit()
            print("Tables created successfully")
        except (Exception, psycopg2.Error) as error:
            print("Error while creating tables", error)

    def insert_stock_data(self, df, symbol):
        engine = create_engine(f'postgresql://{self.db_config["user"]}:{self.db_config["password"]}@{
                               self.db_config["host"]}:{self.db_config["port"]}/{self.db_config["database"]}')
        df['symbol'] = symbol
        df.to_sql('stock_data', engine, if_exists='append', index=False)
        print(f"Inserted stock data for {symbol}")

    def insert_trading_signals(self, df, symbol, strategy):
        engine = create_engine(f'postgresql://{self.db_config["user"]}:{self.db_config["password"]}@{
                               self.db_config["host"]}:{self.db_config["port"]}/{self.db_config["database"]}')
        df['symbol'] = symbol
        df['strategy'] = strategy
        df.to_sql('trading_signals', engine, if_exists='append', index=False)
        print(f"Inserted trading signals for {
              symbol} using {strategy} strategy")

    def insert_performance_metrics(self, metrics, strategy, start_date, end_date):
        query = sql.SQL("""
            INSERT INTO performance_metrics (strategy, start_date, end_date, total_return, sharpe_ratio, max_drawdown)
            VALUES (%s, %s, %s, %s, %s, %s)
        """)
        self.cursor.execute(query, (strategy, start_date, end_date,
                            metrics['Total Return'], metrics['Sharpe Ratio'], metrics['Max Drawdown']))
        self.conn.commit()
        print(f"Inserted performance metrics for {strategy}")

    def fetch_stock_data(self, symbol, start_date, end_date):
        query = sql.SQL("""
            SELECT * FROM stock_data
            WHERE symbol = %s AND date BETWEEN %s AND %s
            ORDER BY date
        """)
        self.cursor.execute(query, (symbol, start_date, end_date))
        columns = [desc[0] for desc in self.cursor.description]
        return pd.DataFrame(self.cursor.fetchall(), columns=columns)

    def fetch_trading_signals(self, symbol, strategy, start_date, end_date):
        query = sql.SQL("""
            SELECT * FROM trading_signals
            WHERE symbol = %s AND strategy = %s AND date BETWEEN %s AND %s
            ORDER BY date
        """)
        self.cursor.execute(query, (symbol, strategy, start_date, end_date))
        columns = [desc[0] for desc in self.cursor.description]
        return pd.DataFrame(self.cursor.fetchall(), columns=columns)

    def fetch_performance_metrics(self, strategy, start_date, end_date):
        query = sql.SQL("""
            SELECT * FROM performance_metrics
            WHERE strategy = %s AND start_date >= %s AND end_date <= %s
        """)
        self.cursor.execute(query, (strategy, start_date, end_date))
        columns = [desc[0] for desc in self.cursor.description]
        return pd.DataFrame(self.cursor.fetchall(), columns=columns)
