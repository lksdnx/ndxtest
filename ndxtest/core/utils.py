"""This module contains various functions, in particular for updating the data library and the histfile.

Imports
-------
os
time
datetime as dt
pandas as pd
pandas.io.formats.excel as pde
numpy as np
copy_tree from distutils.dir_util
DATA_PATH from ndxtest

Functions
---------
constituents(start_date, end_date, lag, data_path=DATA_PATH):
    This function is called during initialization of a BackTest instance. Users of the package do not need it.
download_batch(symbol_list, period="5y", data_path=DATA_PATH):
    Downloads price data from https://finance.yahoo.com/ in the right format.
update_lib(index_symbol='^GSPC', period="3mo", period_first_download="5y", new_entries=0, symbols=None, data_path=DATA_PATH):
    Updates the entire data library.
histfile_rename_symbol(old, new, data_path=DATA_PATH):
    Renames a specific symbol in the histfile.
histfile_new_entry(action, symbol, date, data_path=DATA_PATH):
    Creates a new entry in the histfile with a symbol and the date on which the symbol was added to/removed from the index.

For more information please refer to the docstrings of the functions as well as the online documentation.
"""
import os
import time
import datetime as dt
import pandas as pd
import pandas.io.formats.excel as pde
import numpy as np
import yfinance as yf
from distutils.dir_util import copy_tree


def constituents(start_date, end_date, lag, data_path=''):
    """Returns a dict of symbols and date ranges of inclusion in the index between the start_date and the end_date.

    This function is called during initialization of a new instance of the BackTest class. It's purpose is to read from
    the histfile the date ranges of inclusion in the index for each symbol between the start_date and the end_date. It
    handles special cases such as symbols that entered and left the index repeated times between start_date and end_date

    Parameters
    ----------
    start_date: datetime.datetime Object
        The start date of the backtest period.
    end_date: datetime.datetime Object
        The end date of the backtest period.
    lag: datetime.timedelta Object
        A timedelta that is added in front of the start_date. This is necessary for calculating indicators that depend
        on previous price data such as moving averages and many others.
    data_path: str
        The string should represent the absolute path to where the `data` folder is stored.

    Returns
    -------
    A nested dictionary as follows:
        {symbol: {'dr': pandas.daterange Object representing the daterange of inclusion in the index,
                  'edr': pandas.daterange Object representing 'dr' plus the lag period},
         symbol2: ... }

    For more information please refer to the docstrings of the BackTest class and its methods as well as the online
    documentation.
    """
    hist = pd.read_excel(data_path + 'lib\\^HIST.xlsx', index_col='date', parse_dates=[0])

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


def download_batch(symbol_list, period='5y', data_path=''):
    """This function downloads price data for new symbols from https://finance.yahoo.com/ and performs some formatting.

    Parameters
    ----------
    symbol_list: list
        A list containing the currently valid ticker symbols to download. For example ['AAPL', 'AMZN', 'META']
    period: str, default='5y'
        The time period of data that shall be downloaded. Valid periods are '1d', '5d', '1mo', '3mo', '6mo',
        '1y', '2y', '5y', '10y', 'ytd' and 'max'. It is recommended to download at least 2 years of price data
        when a symbol is added to the index for the first time. This is important for calculation of indicators
        like 200 period moving averages that require a lot of trailing price data.
    data_path: str
        The string should represent the absolute path to where the `data` folder is stored.

    Returns
    -------
    None but creates a new folder data\\downloaded_YYYY-MM-DD\\ where downloaded .csv files for each symbol are stored.

    For additional information please refer to the online documentation.
    """

    os.mkdir(data_path + f'downloaded_{str(dt.datetime.now())[:10]}\\')
    for symbol in symbol_list:
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
                df.to_csv(data_path + f'data\\downloaded_{str(dt.datetime.now())[:10]}\\' + f'{symbol}.csv')

        except KeyError or ValueError or AttributeError:
            print(f'Error processing {symbol}... (may be delisted)')


def update_lib(index_symbol='^GSPC', period='3mo', period_first_download='5y', new_entries=0, symbols=None, data_path=''):
    """This function updates all active symbols in data\\lib\\.

    Active symbols are read from the from the 'histfile'. The histfile as of now has to be updated manually
    by checking for updates to the index on 'https://www.spglobal.com/spdji/en/indices/equity/sp-500/#news-research'.

    Before updating the contents of data\\lib\\, the function creates a backup folder: data\\lib_backup_YYYY-MM-DD\\.
    The function downloads the missing historic price data between the last update (newest record in ^GSPC.csv) and
    today from yahoo finance (using the yfinance package). It performs some checks on the way. If a symbol had a stock
    split between the last update and the current one, the existing data is processed accordingly before appending the
    new records.

    Parameters
    ----------
    index_symbol: str, default='^GSPC'
        '^GSPC' is the ticker symbol of the S&P 500 index on https://finance.yahoo.com/
    period: str, default='3mo'
        The period for which new price data is downloaded, if the last update is more than 3 months back, increase
        accordingly. Valid periods are '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd' and 'max'. Be
        aware that downloading large amounts of data for many symbols can lead to restricted access to the yahoo
        finance API on the way. In this case, the process of updating the library will take some time longer.
    period_first_download: str, default='5y'
        Like period but for symbols with no existing .csv file. (Usually, symbols that were newly added to the index)
    new_entries: int, default=0
        The update_lib function only updates symbols that are currently included in the index, i.e. all symbol listed
        in the newest record in the histfile (the last row). If new entries are added to the histfile BEFORE running
        update_lib (because of index inclusions or exclusions), the last row in the histfile will not contain all
        symbols that need updating. If this is the case, increase new_entries by 1 for each entry that was added.
    symbols: str, default=None
        Here, you can provide a list of symbols if your intention is to only update price data for specific symbols.
        E.g. ['AAPL', 'AMZN'].
    data_path: str
        String that represents the absolute path to where the `data` folder is stored.

    Returns
    -------
    None but updates the price data stored in data\\lib\\.

    For more information on how to maintain the data library and histfile for the ndxtest package, please refer to
    the online documentation.
    """

    from_dir = data_path + 'data\\lib\\'
    to_dir = data_path + f'data\\lib_backup_{str(dt.datetime.now())[:10]}\\'
    copy_tree(from_dir, to_dir)  # a safety backup of the lib folder is generated prior to updating

    if symbols is None:
        hist = pd.read_excel(data_path + 'data\\lib\\^HIST.xlsx')
        activesymbols = sorted(list(set(','.join(hist.symbols.values[-(new_entries + 1):]).split(','))))
        # all symbols within 'last_rows' of the histfile are updated
    else:
        activesymbols = symbols

    lib = [symbol[:-4] for symbol in os.listdir(data_path + 'data\\lib\\') if symbol.endswith('.csv')]
    activesymbols.insert(0, index_symbol)
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
                    df.to_csv(data_path + f'data\\lib\\{symbol}.csv')

            except KeyError or ValueError or AttributeError:
                print(f'Error processing {symbol}... (may be delisted)')

        else:  # the symbol already has a .csv file in //lib
            main = pd.read_csv(data_path + f'data\\lib\\{symbol}.csv', index_col='date', parse_dates=[0])
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
                        print(f'{index_symbol} is used as a reference.')
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
    return None


def histfile_rename_symbol(old, new, data_path=''):
    """Renames a specific symbol in the histfile.

    Parameters
    ----------
    old: str
        Old name of the symbol, e.g. 'FB'
    new: str
        New name of the symbol, e.g. 'META'
    data_path: str
        String that represents the absolute path to where the `data` folder is stored.

    Returns
    -------
    None but overrides the histfile, renaming a specific symbol.

    For more information on how to maintain the histfile please refer to the online documentation.
    """

    pde.ExcelFormatter.header_style = None
    hist = pd.read_excel(data_path + 'data\\lib\\^HIST.xlsx', index_col='date', parse_dates=[0])
    data = []
    for row in hist['symbols']:
        row = row.split(',')
        if old in row:
            i = row.index(old)
            new_row = row[:i] + [new] + row[i+1:]
            data.append(','.join(sorted(new_row)))
        else:
            data.append(','.join(sorted(row)))
    hist['symbols'] = data

    hist['added'].replace(old, new, inplace=True)
    hist['removed'].replace(old, new, inplace=True)

    hist.to_excel(data_path + 'data\\lib\\^HIST.xlsx')
    return None


def histfile_new_entry(action, symbol, date, data_path=''):
    """Creates a new entry in the histfile with a symbol and the date on which it was added to/removed from the index.

    Parameters
    ----------
    action: str
        Accepts 'add' for adding a symbol, or 'remove' for removing a symbol.
    symbol: str
        Name of the new symbol, e.g. 'TSLA'
    date: str of format 'YYYY-MM-DD' or datetime.datetime Object
        The date on which the symbol was added to/removed from the index.
    data_path: str
        String that represents the absolute path to where the `data` folder is stored.

    Returns
    -------
    None but overrides the current histfile, adding a new entry.

    For more information on how to maintain the histfile please refer to the online documentation.
    """

    if (action != 'add') and (action != 'remove'):
        raise ValueError("action must be either 'add' or 'remove'")

    if isinstance(date, str):
        date = dt.datetime.strptime(date, '%Y-%m-%d')
    elif isinstance(date, dt.datetime):
        pass
    else:
        raise TypeError("date_added must be a string formatted 'YYYY-MM-DD' or a datetime.datetime object!")

    pde.ExcelFormatter.header_style = None
    hist = pd.read_excel(data_path + 'data\\lib\\^HIST.xlsx', index_col='date', parse_dates=[0])

    last_row = hist.symbols.values[-1:][0].split(',')

    if action == 'add':
        last_row.append(symbol)
        new_row = ','.join(sorted(last_row))
    else:
        last_row.remove(symbol)
        new_row = ','.join(sorted(last_row))  # will create an Error if symbol not in last row

    new_entry = pd.DataFrame(data={'added': symbol if action == 'add' else '',
                                   'removed': symbol if action == 'remove' else '',
                                   'symbols': new_row},
                             index=[date])

    hist = pd.concat([hist, new_entry])
    hist.index.name = 'date'
    hist.to_excel(data_path + 'data\\lib\\^HIST.xlsx')
    return None
