import pandas as pd
import numpy as np
from pathlib import Path
import sqlalchemy as sql

# create database connection
dp_password = 'stockdata'
engine = sql.create_engine(f'postgresql://postgres:{dp_password}@localhost/stockdata')

local_data_path = Path(f'../data/lib')
symbols = [str(file)[len(f'data\\lib')+1:-len('.csv')] for file in local_data_path.glob(f'*.csv')]


def create_prices_table(ticker_symbol='AAPL'):

    df = pd.read_csv(f'data\\lib\\{ticker_symbol}.csv')
    df['date'] = pd.to_datetime(df['date'])
    df['updated'] = pd.to_datetime('now')
    df.to_sql('daily_prices', engine, if_exists='replace', index=False)
    query = """ALTER TABLE daily_prices ADD PRIMARY KEY (symbol, date)"""
    engine.execute(query)


def fill_db(symbol_list):

    for ticker_symbol in symbol_list:
        print(f'inserting {ticker_symbol}')
        df = pd.read_csv(f'data\\lib\\{ticker_symbol}.csv')
        df['date'] = pd.to_datetime(df.date)
        df['updated'] = pd.to_datetime('now')
        insert_init = """INSERT INTO daily_prices (date, symbol, open, high, low, close, volume, updated) VALUES """
        vals = ",".join([f"('{row.date}','{ticker_symbol}',{row.open},{row.high},{row.low},{row.close},{row.volume},'{row.updated}')" for date, row in df.iterrows()])
        insert_end = """ ON CONFLICT (symbol, date) DO UPDATE 
        SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume,
        updated = EXCLUDED.updated;"""

        query = insert_init + vals + insert_end

        engine.execute(query)
