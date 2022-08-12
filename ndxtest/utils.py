""""""
from collections import OrderedDict
import datetime as dt
from distutils.dir_util import copy_tree
import functools
import numpy as np
import os
import pandas as pd
import pandas.io.formats.excel as pde
import time
import yfinance as yf


def constituents(start_date, end_date, lag, data_path):
    """For each symbol, extracts the date ranges of index inclusion between the start and end date from the histfile.

    Is called within the `import_data()` method of :class:`ndxtest.backtest.BackTest`. `edr` refers to the extended date
    range including the lag.

    :param datetime.datetime start_date: The start date of the backtest period.
    :param datetime.datetime end_date: The end date of the backtest period.
    :param datetime.timedelta lag: A trailing timedelta added to the start_date. Necessary for calculating lagging indicators such as moving averages.
    :param str data_path: Has to represent the absolute path to where the lib folder is stored.

    :returns: A nested dictionary as follows: {symbol: {`dr`: pandas.daterange, `edr`: pandas.daterange}, ...}
    :rtype: dict
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


def connect(data_path, gspc_strict=False, hist_strict=False):
    """Performs checks on the provided `data_path`.

    The `data_path` has to be either the parent directory containing the `data` folder or
    the directory of the `data` folder itself. The `data` folder has to contain the `lib`
    folder.

    :param str data_path: The `data` directory or its parent directory.
    :param bool, default=False, gspc_strict: If True, '^GSPC.csv' has to be in the `lib` folder.
    :param bool, default=False, hist_strict: If True, '^HIST.xlsx' has to be in the `lib` folder.

    :raises: ValueError if data_path was incorrect.

    :returns: The data_path if it has passed the checks.
    :rtype: str
    """

    if not isinstance(data_path, str):
        raise TypeError('data_path was not of type str')

    if data_path.endswith('\\'):
        pass
    else:
        data_path += '\\'

    if data_path.endswith('data\\'):
        pass
    elif 'data' in os.listdir(data_path):
        data_path += 'data\\'
    else:
        raise ValueError("The provided path does not end with and does not contain data\\.")

    if 'lib' not in os.listdir(data_path):
        raise ValueError("data\\ found, but folder lib\\ in data\\.")

    if gspc_strict:
        if '^GSPC.csv' not in os.listdir(data_path + 'lib\\'):
            raise FileNotFoundError("data\\lib\\ found, but '^GSPC.csv' is missing.")

    if hist_strict:
        if '^HIST.xlsx' not in os.listdir(data_path + 'lib\\'):
            raise FileNotFoundError("data\\lib\\ found, but '^HIST.xlsx' is missing.")

    return data_path


def timeit_decorator(func):
    """This decorator prints the time needed for function execution to the console.

    :param function func: Any function.
    """
    @functools.wraps(func)
    def wrapper_func(*args, **kwargs):
        first = f"Finished {func.__name__}..."
        j = 40 - len(func.__name__)
        indent = ''
        for i in range(0, j):
            indent += ' '
        t0 = time.time()
        func(*args, **kwargs)
        t1 = time.time()
        delta = (t1 - t0).__round__(2)
        print(first + f"{indent}...{delta} sec elapsed")
        return None
    return wrapper_func


class Portfolio:
    """The Portfolio class simulates a portfolio. It is a code container not meaningful for users of ndxtest.

    Instances of the BackTest class instantiate a portfolio from this class. Within the .run_backtest() method of the
    BackTest instance the portfolio and its methods help to structure the codebase and make it more readable.
    The Portfolio class may be shifted to ndxtest.core.utils in future releases.

    :ivar int max_positions: Maximum number of positions (slots). Initial value: param 'max_positions' of __init__
    :ivar float cash: Available unrestricted cash in the balance. Initial value: param 'initial_equity' of __init__
    :ivar float commission: Commission paid upon entering/exiting positions (e.g. 0.01 -> 1%). Initial value: param 'commission' of __init__
    :ivar float invested_capital: Current market value of the invested capital. Initial value: 0
    :ivar float commission_paid: Cumulative commissions paid during the backtest. Initial value: 0
    :ivar int number_of_trades: Cumulative number of trades executed. Is incremented when a position is entered but not when closed/covered. Initial value: 0
    :ivar dict cash_from_short_positions: The 'restricted' cash that is part of the balance as a result of entering short positions. Initial value: {}
    :ivar dict long_positions: Contains information about the current long positions. Initial value: {}
    :ivar dict short_positions: Contains information about the current short positions. Initial value: {}
    :ivar OrderedDict logdict:  Contains information about closed positions. Used to generate a report on the backtest. Initial value: OrderedDict()
    """

    def __init__(self, max_positions, initial_equity, commission):
        """Constructor method.

        :param int max_positions: Maximum number of positions/slots of the portfolio.
        :param float initial_equity: The initial capital to start with.
        :param float commission: Commission paid upon entering/exiting positions (e.g. 0.01 -> 1%).

        """

        self.max_positions = max_positions
        self.cash = initial_equity
        self.commission = commission
        self.invested_capital, self.commission_paid, self.number_of_trades = 0, 0, 0
        self.cash_from_short_positions = {}
        self.long_positions, self.short_positions = {}, {}
        self.logdict = OrderedDict()

    def positions(self):
        """
        :returns: The union of ticker symbols that currently have open long or short positions.
        :rtype: set
        """
        return set(list(self.long_positions.keys()) + list(self.short_positions.keys()))

    def number_of_positions(self):
        """
        :returns: The number of currently filled positions.
        :rtype: int
        """

        return len(self.positions())

    def free_slot(self):
        """
        :returns: True when there is a free slot in the portfolio or False if there is none.
        :rtype: bool
        """

        return True if self.number_of_positions() < self.max_positions else False

    def current_exposure(self):
        """
        :returns: A dict with some info on current positions.
        :rtype: dict
        """

        return {'n_long': len(self.long_positions.keys()),
                'n_short': len(self.short_positions.keys()),
                'n_free': self.max_positions - self.number_of_positions(),
                'net_exposure': len(self.long_positions.keys()) - len(self.short_positions.keys())}

    def restricted_cash(self):
        """Returns the amount of 'restricted' cash, resulting from entering short positions. This cash is not considered
        for calculating the position size when entering new positions.

        :returns: The amount of 'restricted' cash.
        :rtype: float
        """

        return sum(self.cash_from_short_positions.values())

    def calculate_nshares(self, entry_price):
        """Calculates the number of shares to buy/sell based on free slots, share price and cash balance.

        :returns: Number of shares to buy/sell.
        :rtype: int
        """

        return (((self.cash - self.restricted_cash()) /
                 (self.max_positions - self.number_of_positions())) / entry_price).__floor__()

    def describe(self, date):
        """Prints a summary of some Portfolio stats to the console.

        :returns: None
        :rtype: NoneType
        """

        print(f'Stats of {self} as of {date}:')
        print(f'Number of Positions: {self.number_of_positions()}. '
              f'Long: {list(self.long_positions.keys())}, '
              f'Short: {list(self.short_positions.keys())}')
        print(f'Cash: {self.cash.__floor__()}. Cash from shorts: {self.restricted_cash().__floor__()}. '
              f'Invested Capital: {self.invested_capital.__floor__()}.')
        print(f'Total Market Value: {(self.invested_capital + self.cash + self.restricted_cash()).__floor__()}')
        print('\n')

    def create_log_entry(self, entry_data=None, exit_data=None):
        """Creates a record in the instance variable `logdict` upon entering or exiting positions.

        :param dict, default=None entry_data: Contains some data connected with the entry into the position.
        :param dict, default=None exit_data: Contains some data connected with the exit from the position.

        :returns: A dict with aggregated information about the position. Used to create a log record.
        :rtype: dict
        """

        entry_comm = abs(entry_data['entry_price'] * entry_data['nshares'] * self.commission)
        duration, exit_comm, pnl, p_pnl = 0, 0, 0, 0
        if exit_data:
            duration = (exit_data['exit_date'] - entry_data['entry_date']).days
            exit_comm = abs(exit_data['exit_price'] * entry_data['nshares'] * self.commission)
            pnl = (exit_data['exit_price'] - entry_data['entry_price']) * entry_data['nshares']
            p_pnl = ((pnl / (entry_data['entry_price'] * entry_data['nshares'])) * 100)
        d = {'id': entry_data['id'],
             'entry_date': entry_data['entry_date'],
             'entry_weekday': entry_data['entry_date'].isoweekday(),
             'exit_date': exit_data['exit_date'] if exit_data else 0,
             'exit_weekday': exit_data['exit_date'].isoweekday() if exit_data else 0,
             'duration': duration,
             'symbol': entry_data['symbol'],
             'long': 1 if entry_data['signal'] == 1 else 0,
             'short': 1 if entry_data['signal'] == -1 else 0,
             'nshares': entry_data['nshares'],
             'entry_price': entry_data['entry_price'],
             'entry_score': entry_data['entry_score'],
             'entry_value': entry_data['entry_price'] * entry_data['nshares'],
             'exit_price': exit_data['exit_price'] if exit_data else 0,
             'exit_score': exit_data['exit_score'] if exit_data else 0,
             'exit_value': exit_data['exit_price'] * entry_data['nshares'] if exit_data else 0,
             'entry_comm': -entry_comm if not exit_data else 0,
             'exit_comm': -exit_comm,
             '$change': pnl,
             '%change': p_pnl,
             'profitable': 1 if pnl > 0 else 0,
             'p.positions': f'l: {len(list(self.long_positions.keys()))} s: {len(list(self.short_positions.keys()))}',
             'p.cash': self.cash,
             'p.res_cash': self.restricted_cash(),
             'p.invested_capital': self.invested_capital,
             'market_value': (self.invested_capital + self.cash + self.restricted_cash())}
        return d

    def create_log_df(self):
        """
        :returns: A pd.DataFrame, created from the instance variable `logdict`.
        :rtype: pd.DataFrame
        """

        return pd.DataFrame.from_dict(data=self.logdict, orient='index').round(2)

    def long(self, data):  # closing short position has priority
        """Creates a long, or closes a short position in the portfolio. Closing an existing short position has precedence
        over creating a new long position.

        :param dict data: Contains information regarding the entered/covered position.

        :returns: None
        :rtype: NoneType
        """

        if data['symbol'] in self.short_positions.keys():
            nshares = self.short_positions[data['symbol']]['nshares']
            entry_value = self.cash_from_short_positions[data['symbol']]
            current_value = -nshares * data['exit_price']

            self.cash += entry_value - current_value  # subtracting the P/L on the cash side
            self.cash -= abs(self.commission * nshares * data['exit_price'])
            self.commission_paid += abs(self.commission * nshares * data['exit_price'])
            self.invested_capital -= (nshares * data['exit_price']) - (entry_value - current_value)
            del self.cash_from_short_positions[data['symbol']]

            self.logdict[f"{str(data['exit_date'])[:10]}_{data['symbol']}"] = \
                self.create_log_entry(entry_data=self.short_positions[data['symbol']], exit_data=data)

            del self.short_positions[data['symbol']]
        else:  # entering a new long position
            if self.free_slot():
                self.number_of_trades += 1
                data['id'] = self.number_of_trades
                self.long_positions[data['symbol']] = data

                self.cash -= (data['nshares'] * data['entry_price']) + (
                        self.commission * data['nshares'] * data['entry_price'])
                self.commission_paid += self.commission * data['nshares'] * data['entry_price']
                self.invested_capital += data['nshares'] * data['entry_price']
                self.logdict[f"{str(data['entry_date'])[:10]}_{data['symbol']}"] = \
                    self.create_log_entry(entry_data=self.long_positions[data['symbol']])

    def short(self, data):  # closing long position has priority
        """Creates a short, or closes a long position in the portfolio. Closing an existing long position has precedence
        over creating a new short position.

        :param dict data: Contains information regarding the entered/closed position.

        :returns: None
        :rtype: NoneType
        """

        if data['symbol'] in self.long_positions.keys():
            entry_price = self.long_positions[data['symbol']]['entry_price']
            nshares = self.long_positions[data['symbol']]['nshares']

            self.cash += (nshares * data['exit_price']) - (self.commission * nshares * data['exit_price'])
            self.commission_paid += self.commission * nshares * data['exit_price']
            self.invested_capital -= nshares * (entry_price - data['exit_price'])  # subtracting the P/L
            self.invested_capital -= nshares * data['exit_price']
            self.logdict[f"{str(data['exit_date'])[:10]}_{data['symbol']}"] = \
                self.create_log_entry(entry_data=self.long_positions[data['symbol']], exit_data=data)
            del self.long_positions[data['symbol']]
        else:  # entering a new short position
            if self.free_slot():
                self.number_of_trades += 1
                data['id'] = self.number_of_trades
                self.short_positions[data['symbol']] = data

                self.cash -= abs(self.commission * data['nshares'] * data['entry_price'])
                self.cash_from_short_positions[data['symbol']] = -data['nshares'] * data['entry_price']
                self.commission_paid += self.commission * -data['nshares'] * data['entry_price']
                self.invested_capital += data['nshares'] * data['entry_price']

                self.create_log_entry(entry_data=self.short_positions[data['symbol']])


class LibManager:
    """This class is the main tool for maintaining and updating the data library.

    :ivar str data_path: The path to the data.
    :ivar list symbols_in_lib: A list of all symbols found in \\lib.
    """

    def __init__(self, data_path):
        """Constructor method.

        :param str data_path: Has to represent directory to the `data` folder, or the `data` folders parent directory.
        """

        self.data_path = connect(data_path)
        self.symbols_in_lib = [symbol[:-4] for symbol in os.listdir(self.data_path + 'lib\\') if symbol.endswith('.csv')]

    def download_batch(self, symbol_list: list, period='5y'):
        """Downloads price data for new symbols from https://finance.yahoo.com/ and performs some formatting.

        Creates a new folder data\\downloaded_YYYY-MM-DD\\ where downloaded .csv files are saved.

        :param list symbol_list: Valid ticker symbols to download. For example: ['AAPL', 'AMZN', 'META']
        :param str, default='5y' period: Time period of data to download. Valid values are: '1d', '5d', '1mo', '3mo', '6mo',
            '1y', '2y', '5y', '10y', 'ytd' and 'max'. It is recommended to download at least 2 years of price data
            when a symbol is added to the index for the first time. This is important for calculation of indicators
            like 200 period moving averages that require a lot of trailing price data.

        :returns: None
        :rtype: NoneType
        """

        if not isinstance(symbol_list, list):
            raise TypeError("Parameter `symbol_list` must be of type `list`.")

        if not isinstance(period, str):
            raise TypeError("Parameter `period` must be of type `str`.")

        if period not in ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']:
            raise ValueError("Parameter `period` must be either '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd' or 'max'.")

        if f'downloaded_{str(dt.datetime.now())[:10]}' in os.listdir(self.data_path):
            pass
        else:
            os.mkdir(self.data_path + f'downloaded_{str(dt.datetime.now())[:10]}\\')

        for symbol in symbol_list:
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

                if 0 in df[['open', 'high', 'low', 'close']].values:
                    print(f'Warning: {symbol} contained missing values in price data.')

                print(f"Downloaded {len(df.index)} records for {symbol}.")

                if len(df.index) > 1:
                    df.to_csv(self.data_path + f'downloaded_{str(dt.datetime.now())[:10]}\\' + f'{symbol}.csv')

            except KeyError or ValueError or AttributeError:
                print(f'Error processing {symbol}... (symbol may be delisted or a name change may have occurred)')

    def lib_update(self, index_symbol='^GSPC', period='3mo', period_first_download='5y', new_entries=0, symbols=None):
        """This function updates all `active` symbols in data\\lib\\.

        Active symbols are read from the from the `histfile`. As of now, the histfile has to be updated manually
        by checking for updates to the index on 'https://www.spglobal.com/spdji/en/indices/equity/sp-500/#news-research'.

        Before updating the contents of data\\lib\\, a backup folder: data\\lib_backup_YYYY-MM-DD\\ is created.
        Missing historic price data between the last update (newest record in ^GSPC.csv) and  is downloaded from yahoo finance
        with the help of the the yfinance package. If a symbol had a stock split between the last and the current update,
        the existing data is processed accordingly before appending the new records.

        :param str, default='^GSPC' index_symbol: '^GSPC' is the ticker symbol of the S&P 500 index on https://finance.yahoo.com/
        :param str, default='3mo' period: The period for which new price data is downloaded, if the last update is more than 3 months back, increase
            accordingly. Valid periods are '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd' and 'max'. Be
            aware that downloading large amounts of data for many symbols can lead to restricted access to the yahoo
            finance API on the way. In this case, the process of updating the library will take some time longer.
        :param str, default='5y' period_first_download: Like `period` but for symbols with no existing .csv file.
            Usually, symbols that were newly added to the index
        :param int, default=0 new_entries: The update_lib function only updates symbols that are currently included in the index, i.e. all symbol listed
            in the newest record in the `histfile`. If new entries are added to the histfile BEFORE running
            update_lib, the last row in the histfile will not contain all symbols that need updating. If this is the case,
            increase `new_entries` by 1 for each entry that was added.
        :param list or str, default=None symbols: Here, you can provide a list of symbols if your intention is to only update price data for specific symbols.
            E.g. ['AAPL', 'AMZN'].

        :returns: None
        :rtype: NoneType

        For more information on how to maintain the data library and histfile for the ndxtest package, please refer to
        the user manual.
        """

        self.data_path = connect(self.data_path, gspc_strict=True, hist_strict=True)

        from_dir = self.data_path + 'lib\\'
        to_dir = self.data_path + f'\\lib_backup_{str(dt.datetime.now())[:10]}\\'
        copy_tree(from_dir, to_dir)  # a safety backup of the lib folder is generated prior to updating

        if symbols is None:
            hist = pd.read_excel(self.data_path + 'lib\\^HIST.xlsx')
            activesymbols = sorted(list(set(','.join(hist.symbols.values[-(new_entries + 1):]).split(','))))
            # all symbols within 'last_rows' of the histfile are updated
        else:
            if isinstance(symbols, list):
                activesymbols = symbols
            elif isinstance(symbols, str):
                activesymbols = [symbols]
            else:
                raise TypeError("Parameter `symbols` must be of type `list` or `str`.")

        activesymbols.insert(0, index_symbol)
        approved = False
        breaker = False
        reference_index = None

        for symbol in activesymbols:

            if symbol not in self.symbols_in_lib:  # the symbol has no .csv file in //lib
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
                        df.to_csv(self.data_path + f'lib\\{symbol}.csv')

                except KeyError or ValueError or AttributeError:
                    print(f'Error processing {symbol}... (symbol may be delisted or a name change may have occurred)')

            else:  # the symbol already has a .csv file in //lib
                main = pd.read_csv(self.data_path + f'lib\\{symbol}.csv', index_col='date', parse_dates=[0])
                time.sleep(.3)
                try:
                    data = yf.Ticker(symbol)
                    df = data.history(period=period)
                    df.insert(0, 'date', df.index)
                    df.set_index('date', inplace=True)

                    try:
                        df.columns = ['open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits']
                    except ValueError:
                        print(f'Error processing {symbol}... (symbol may be delisted or a name change may have occurred)')

                    df.insert(0, 'symbol', symbol)

                    if 0 in df[['open', 'high', 'low', 'close']].values:
                        print(f'Warning: {symbol} contained missing values in price data.')
                        approved = False

                    diff = df.index.difference(main.index)
                    df = df.loc[diff]

                    if any(df.stock_splits.values[:]):
                        s = float(df.loc[df.stock_splits.values[:] != 0, 'stock_splits'])  # s = split_factor
                        print(f'Processed split by factor of {s} for {symbol}.')
                        main = main.divide([None, s, s, s, s, 1 / s, s, 1])
                        main['symbol'] = symbol

                    while not approved:
                        if reference_index is None:
                            print(f'{index_symbol} is used as a reference.')
                            print(f'The last updated was performed on {main.index[-1]}.')
                            print(f'Downloaded data contains {len(df.index)} new records.')
                            print(f'New records: {list(df.index)}.')
                            print(
                                f'Caution, do only perform this updating routine on weekends or after US market close.')
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
                        pd.concat([main, df]).to_csv(f'data\\lib\\{symbol}.csv')
                    else:
                        print(f'Downloaded data for {symbol} contained following reference_index:')
                        print(df.index)
                        print(f'This does not match the reference index:')
                        print(reference_index)
                        if input(f'Append nevertheless? Y/N:').upper() == 'Y':
                            print(f'Appending {len(df.index)} records to {symbol}...')
                            pd.concat([main, df]).to_csv(f'data\\lib\\{symbol}.csv')
                        if input(f'Continue with updating routine? Y/N:').upper() == 'N':
                            breaker = True

                except KeyError or ValueError or AttributeError:
                    print(f'Error processing {symbol}... (symbol may be delisted or a name change may have occurred)')
        return None

    def lib_rename_symbol(self, old: str, new: str):
        """Renames a symbol in the library and the `histfile`.

        :param str old: Old name of the symbol, e.g. 'FB'
        :param str new: New name of the symbol, e.g. 'META'

        :returns: None
        :rtype: NoneType
        """

        if not isinstance(old, str) and isinstance(new, str):
            raise TypeError("Parameters `old` and `new` must be of type `str`.")

        self.data_path = connect(self.data_path, hist_strict=True)

        # renaming in the histfile
        pde.ExcelFormatter.header_style = None
        hist = pd.read_excel(self.data_path + 'lib\\^HIST.xlsx', index_col='date', parse_dates=[0])
        data = []
        for row in hist['symbols']:
            row = row.split(',')
            if old in row:
                i = row.index(old)
                new_row = row[:i] + [new] + row[i + 1:]
                data.append(','.join(sorted(new_row)))
            else:
                data.append(','.join(sorted(row)))
        hist['symbols'] = data

        hist['added'].replace(old, new, inplace=True)
        hist['removed'].replace(old, new, inplace=True)

        hist.to_excel(self.data_path + 'lib\\^HIST.xlsx')

        # renaming in the library
        file = pd.read_csv(self.data_path + f'lib\\{old}.csv', index_col='date', parse_dates=[0])
        file['symbol'] = new
        file.to_csv(f'data\\lib\\{new}.csv')
        if new != old:
            os.remove(self.data_path + f'lib\\{old}.csv')
        return None

    def histfile_new_entry(self, action: str, symbol: str, date: str or dt.datetime):
        """Creates a new entry in the `histfile` with a symbol and the date of addition to/removal from the index.

        :param str action: Accepts 'add' for adding a symbol, or 'remove' for removing a symbol.
        :param str symbol: Name of the new symbol, e.g. 'TSLA'.
        :param str or datetime.datetime date: The date on which the symbol was added to/removed from the index. If a str is provided it must be 'YYYY-MM-DD' formatted.

        :returns: None
        :rtype: NoneType

        For more information on how to maintain the histfile please refer to the user manual.
        """

        self.data_path = connect(self.data_path, hist_strict=True)

        if not isinstance(action, str):
            raise TypeError("Argument `action` must be of type str.")

        if (action != 'add') and (action != 'remove'):
            raise ValueError("Argument `action` must be either 'add' or 'remove'")

        if isinstance(date, str):
            if ('-' not in date) or (len(date) != 10):
                raise ValueError("Argument `date` has to be 'YYYY-MM-DD' formatted if type str is provided.")
            date = dt.datetime.strptime(date, '%Y-%m-%d')
        elif isinstance(date, dt.datetime):
            pass
        else:
            raise TypeError("date_added must be a string formatted 'YYYY-MM-DD' or a datetime.datetime object!")

        pde.ExcelFormatter.header_style = None
        hist = pd.read_excel(self.data_path + 'lib\\^HIST.xlsx', index_col='date', parse_dates=[0])

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
        hist.to_excel(self.data_path + 'lib\\^HIST.xlsx')
        return None


