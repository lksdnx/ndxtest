import os
import time
import datetime as dt
import pandas as pd
import numpy as np
import yfinance as yf
from distutils.dir_util import copy_tree


def constituents(data_path, start_date, end_date, lag):
    """Returns a dictionary of symbols and inclusion date_ranges in the benchmark index
    between the start_date and the end_date."""
    hist = pd.read_excel(f'{data_path}^HIST.xlsx', index_col='date', parse_dates=[0])

    dr = pd.date_range(start_date, end_date)
    edr = pd.date_range(start_date - lag, end_date)
    hist = hist.loc[hist.index.intersection(dr)]

    tickers = []
    for row in hist.symbols.values:
        for s in row.split(','):
            if s not in tickers:
                tickers += [s]

    cons = {k: None for k in tickers}

    added = [s for s in hist.added.values if str(s) != 'nan']
    removed = [s for s in hist.removed.values if str(s) != 'nan']

    multiple = list(set(added).intersection(set(removed)))
    added = [s for s in added if s not in multiple]
    removed = [s for s in removed if s not in multiple]
    whole_period = [s for s in tickers if s not in added and s not in removed and s not in multiple]

    for s in multiple:
        df = hist.loc[(hist['added'] == s) | (hist['removed'] == s)]
        if len(df.index) > 2:
            print(f'Warning: {s} had > 2 index inclusions/removals during time period.')

        if df.loc[df.index[0]].added == s:
            cons[s] = {'dr': pd.date_range(df.index[0], df.index[1]),
                       'edr': pd.date_range(df.index[0] - lag, df.index[1])}
        if df.loc[df.index[0]].removed == s:
            cons[f'{s}'] = {'dr': pd.date_range(start_date, df.index[0]), 'edr': pd.date_range(start_date - lag, df.index[0])}
            cons[f'{s}*'] = {'dr': pd.date_range(df.index[1], end_date), 'edr': pd.date_range(df.index[1] - lag, end_date)}

    for s in whole_period:
        cons[s] = {'dr': dr,
                   'edr': edr}
    for s in added:
        cons[s] = {'dr': pd.date_range(hist.loc[hist.added == s].index[0], end_date),
                   'edr': pd.date_range(hist.loc[hist.added == s].index[0] - lag, end_date)}
        if s in removed:
            cons[s] = {'dr': pd.date_range(hist.loc[hist.added == s].index[0], hist.loc[hist.removed == s].index[0]),
                       'edr': pd.date_range(hist.loc[hist.added == s].index[0] - lag, hist.loc[hist.removed == s].index[0])}
            removed.remove(s)
    for s in removed:
        cons[s] = {'dr': pd.date_range(start_date, hist.loc[hist.removed == s].index[0]),
                   'edr': pd.date_range(start_date - lag, hist.loc[hist.removed == s].index[0])}
    return cons


def ybatchdownload(symbollist, period="5y"):
    """This function performs a batchdownload from yfinance and does some formatting. It saves downloaded data into the
    data folder. Valid periods are ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max", None]"""
    for symbol in symbollist:
        print(f'Loading {symbol}...')
        try:
            data = yf.Ticker(symbol)
            df = data.history(period=period)
            df.insert(0, 'date', df.index)
            df.set_index('date', inplace=True)
            try:
                df.columns = ['open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits']
            except ValueError:
                pass
            df.insert(0, 'symbol', symbol)
            df['open'] = np.float32(df['open'])
            df['high'] = np.float32(df['high'])
            df['low'] = np.float32(df['low'])
            df['close'] = np.float32(df['close'])

            if any(df.stock_splits.values[-100:]):
                print(f'Warning: {symbol} had a stock split in the last 100 trading days.')

            if 0 in df[['open', 'high', 'low', 'close']].values:
                print(f'Warning: {symbol} contained missing values in price data.')

            print(f"{len(df.index)} records downloaded {symbol}.")

            if len(df.index) > 1:
                df.to_csv(f'data\\{symbol}.csv')

        except KeyError or ValueError or AttributeError:
            print(f'Error processing {symbol}... (may be delisted)')


def yupdatelib(benchmark_symbol='^GSPC', histfile='data\\lib\\^HIST.xlsx', last_rows=1,
               period="3mo", period_first_download="2y", symbols=None):
    """This function updates all active symbols in data//lib. Active symbols are read from the
    from the 'histfile' which as of now has to be updated manually by checking for updates to the S&P 500 index
    on the S&P global website: 'https://www.spglobal.com/spdji/en/indices/equity/sp-500/#news-research'.
    The function downloads the missing historic price data between the last update (newest record in ^GSPC.csv) and
    today from yahoo finance (using the yfinance package). It checks for missing price data.
    If a symbol does not yet have a .csv file in the //lib folder e.g. because it was recently added to the index,
    2 years (period_first_download) worth of historic price data will be downloaded by default.
    If a symbol had a stock split between the last update and the current one, the existing data is processed
    accordingly before appending the new records. The 'last_rows' parameter (1 by default) can be increased to download
    price data for symbols that have recently been removed from the index (e.g. with a value of 10, the set of the
    symbols of the last 10 rows of the histfile is updated). Valid values for period/period_first_download are
    ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max", None]"""

    # out = sys.stdout
    # sys.stdout = open(f"data\\log_{str(dt.datetime.now())[:10]}.txt", "w")

    from_dir = "data\\lib\\"
    to_dir = f"data\\lib_backup_{str(dt.datetime.now())[:10]}\\"
    copy_tree(from_dir, to_dir)  # a safety backup of the lib folder is generated prior to updating

    if symbols is None:
        hist = pd.read_excel(histfile)
        activesymbols = sorted(list(set(','.join(hist.symbols.values[-last_rows:]).split(','))))
        # all symbols within 'last_rows' of the histfile are updated
    else:
        activesymbols = symbols

    lib = [symbol[:-4] for symbol in os.listdir('data\\lib\\') if symbol.endswith('.csv')]
    activesymbols.insert(0, benchmark_symbol)
    approved = False
    breaker = False
    reference_index = None

    for symbol in activesymbols:

        if symbol not in lib:  # the symbol has no .csv file in //lib
            print(f"{symbol} not found in lib, downloading {period_first_download}'s of price data...")
            try:
                data = yf.Ticker(symbol)
                df = data.history(period=period_first_download)
                df.insert(0, 'date', df.index)
                df.set_index('date', inplace=True)
                try:
                    df.columns = ['open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits']
                except ValueError:
                    pass
                df.insert(0, 'symbol', symbol)
                df['open'] = np.float32(df['open'])
                df['high'] = np.float32(df['high'])
                df['low'] = np.float32(df['low'])
                df['close'] = np.float32(df['close'])

                if 0 in df[['open', 'high', 'low', 'close']].values:
                    print(f'Warning: {symbol} contained missing values in price data.')

                print(f"{len(df.index)} records downloaded for {symbol}...")

                if len(df.index) > 1:
                    df.to_csv(f'data\\lib\\{symbol}.csv')

            except KeyError or ValueError or AttributeError:
                print(f'Error processing {symbol}... (may be delisted)')

        else:  # the symbol already has a .csv file in //lib
            main = pd.read_csv(f'data\\lib\\{symbol}.csv', index_col='date', parse_dates=[0])
            time.sleep(.3)
            try:
                data = yf.Ticker(symbol)
                df = data.history(period=period)
                df.insert(0, 'date', df.index)
                df.set_index('date', inplace=True)

                try:
                    df.columns = ['open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits']
                except ValueError:
                    print(f'Error processing {symbol}... (may be delisted)')

                df.insert(0, 'symbol', symbol)

                if 0 in df[['open', 'high', 'low', 'close']].values:
                    print(f'Warning: {symbol} contained missing values in price data.')
                    approved = False

                diff = df.index.difference(main.index)
                df = df.loc[diff]

                if any(df.stock_splits.values[:]):
                    s = float(df.loc[df.stock_splits.values[:] != 0, 'stock_splits'])     # s = split_factor
                    print(f'Processed split by factor of {s} for {symbol}.')
                    main = main.divide([None, s, s, s, s, 1 / s, s, 1])
                    main['symbol'] = symbol
                    main['volume'] = main['volume'].astype('int64')

                while not approved:
                    if reference_index is None:
                        print(f'{benchmark_symbol} is used as a reference.')
                        print(f'The last updated was performed on {main.index[-1]}.')
                        print(f'Downloaded data contains {len(df.index)} new records.')
                        print(f'New records: {list(df.index)}.')
                        print(f'Caution, do only perform this updating routine on weekends or after US market close.')
                    if input(f'Proceed appending new records to all files? Y/N:').upper() == 'Y':
                        approved = True
                        reference_index = df.index
                    else:
                        breaker = True
                        break

                if breaker:
                    print('Process terminated by user.')
                    break

                if reference_index.identical(df.index):
                    print(f'Appending {len(df.index)} records to {symbol}...')
                    main = main.append(df)
                    main.to_csv(f'data\\lib\\{symbol}.csv')
                else:
                    print(f'Downloaded data for {symbol} contained following reference_index:')
                    print(df.index)
                    print(f'This does not match the reference index:')
                    print(reference_index)
                    if input(f'Append nevertheless? Y/N:').upper() == 'Y':
                        print(f'Appending {len(df.index)} records to {symbol}...')
                        main = main.append(df)
                        main.to_csv(f'data\\lib\\{symbol}.csv')
                    if input(f'Continue with updating routine? Y/N:').upper() == 'N':
                        breaker = True

            except KeyError or ValueError or AttributeError:
                print(f'Error processing {symbol}... (may be delisted)')

    # sys.stdout.close()
    # sys.stdout = out
    # end of yupdatelib


def histfile_rename_symbol(histfile='data\\lib\\^HIST.xlsx', old=None, new=None):
    hist = pd.read_excel(histfile, index_col='date', parse_dates=[0])
    data = []
    for row in hist.symbols:
        if f'{old}' in row:
            new_row = row.replace(f'{old}', f'{new}')
            new_row = ','.join(sorted(new_row.split(',')))
            data.append(new_row)
        else:
            print('Ticker to replace was not found.')
            data.append(row)
    hist['symbols'] = data
    hist.to_excel('data\\lib\\^HIST.xlsx')
    return None
