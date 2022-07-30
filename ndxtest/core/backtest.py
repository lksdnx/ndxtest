"""This is backtest.py, the main module of the ndxtest package containing the BackTest class.

Classes
-------
BackTest
    The BackTest class is where the bulk of the code of the ndxtest package is located. For a specific date range, it
    imports the data from data\\lib\\. It accepts a Strategy object that can be built with the Strategy class from
    strategy.py. It instantiates a portfolio from the Portfolio class and runs a backtest of the provided strategy.
    It can perform optimization of specific parameters provided within a strategy (comment: not included in the current
    release!). Finally it produces reports in the form of .xlsx and .pdf files.
Portfolio
    The Portfolio class is a code container that is not directly used by users of this package. The instances of the
    BackTest class instantiate a portfolio from this class. Within the .run_backtest() method of the BackTest instance
    the portfolio and its methods help to structure the codebase and make it more readable. The Portfolio class may be
    shifted to ndxtest.core.utils in future releases.

For further information please refer to the class docstrings and to the online documentation.
"""

from pathlib import Path
import os
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import pandas as pd
import decimal
import itertools
from collections import OrderedDict, defaultdict
from ndxtest.core.utils import constituents
import time
import datetime as dt
import mplfinance as mpf
import random
import seaborn as sns
from fpdf import FPDF


class Portfolio:
    """The Portfolio class simulates a portfolio. It is a code container and not directly meaningful for users ndxtest.

    Instances of the BackTest class instantiate a portfolio from this class. Within the .run_backtest() method of the
    BackTest instance the portfolio and its methods help to structure the codebase and make it more readable.
    The Portfolio class may be shifted to ndxtest.core.utils in future releases.

    Attributes
    ----------
    self.max_positions: int
        Maximum number of positions (slots). No more positions are entered when all slots are filled.
    self.cash: float
        Available unrestricted cash in the balance.
    self.commission: float
        Commission that is paid upon entering/exiting positions (e.g. 0.01 would be 1%)
    self.invested_capital: float
        Current market value of the invested capital.
    self.commission_paid: float
        Cumulative commissions paid during the backtest.
    self.number_of_trades: int
        Cumulative number of trades executed. Is incremented when a position is entered but not when closed/covered.
    self.cash_from_short_positions: dict
        The 'restricted' cash that is part of the balance as a result of entering short positions.
    self.long_positions: dict
        A nested dictionary with information about the current long positions.
    self.short_positions: dict
        A nested dictionary with information about the current short positions.
    self.logdict = OrderedDict()
        A dictionary with information about closed positions. Used to generate a report on the entire backtest.

    Methods
    _______
    __init__(self, max_positions, initial_equity, commission):
        Initialization of the portfolio.
    positions(self):
        Returns the set() union of ticker symbols that currently have open long or short positions.
    number_of_positions(self):
        Returns the number of currently filled positions.
    free_slot(self):
        Returns whether there is a free slot for an incremental position (True/False).
    current_exposure(self):
        Returns a dict() with some info on current positions (see docstring of method).
    restricted_cash(self):
        Returns the amount of 'restricted' cash in the portfolio that resulted from entering short positions.
    calculate_nshares(self, entry_price):
        Returns the number of shares to buy/sell based on free slots, share price and unrestricted cash in the portfolio.
    describe(self):
        Prints a summary of some Portfolio stats to the console.
    create_log_entry(self, entry_data=None, exit_data=None):
        Creates an entry to self.logdict upon entering or exiting positions.
    create_log_df(self):
        Creates a pd.DataFrame from self.logdict.
    long(self, data):
        Creates a long, or closes a short position in the portfolio.
    short(self, data):
        Creates a short, or closes a long position in the portfolio.

    For further information please refer to the online documentation.
    """

    def __init__(self, max_positions, initial_equity, commission):
        """Initializes an instance of the Portfolio class.

        Parameters
        ----------
        max_positions: int
            Maximum number of positions/slots of the portfolio.
        initial_equity: float
            The initial capital to start with.
        commission: float
            The amount of commission that is taken for entering and exiting positions.

        Returns
        -------
        None
        """
        self.max_positions = max_positions
        self.cash = initial_equity
        self.commission = commission
        self.invested_capital, self.commission_paid, self.number_of_trades = 0, 0, 0
        self.cash_from_short_positions = {}
        self.long_positions, self.short_positions = {}, {}
        self.logdict = OrderedDict()

    def positions(self):
        """Returns the set() union of ticker symbols that currently have open long or short positions."""
        return set(list(self.long_positions.keys()) + list(self.short_positions.keys()))

    def number_of_positions(self):
        """Returns the number of currently filled positions."""
        return len(self.positions())

    def free_slot(self):
        """Returns whether there is a free slot for an incremental position or not.

        Returns
        -------
        True when there is a free slot in the portfolio
        False when there is no free slot in the portfolio
        """
        return True if self.number_of_positions() < self.max_positions else False

    def current_exposure(self):
        """Returns a dict with some info on current positions.

        The dict has the following entries:
        {'n_long': number of long positions,
         'n_short': number of short positions,
         'n_free': number of free slots,
         'net_exposure': number of long positions - number of short positions}
        """
        return {'n_long': len(self.long_positions.keys()),
                'n_short': len(self.short_positions.keys()),
                'n_free': self.max_positions - self.number_of_positions(),
                'net_exposure': len(self.long_positions.keys()) - len(self.short_positions.keys())}

    def restricted_cash(self):
        """Returns the amount of 'restricted' cash in the portfolio that resulted from entering short positions.

        This cash is not considered for calculating the position size when entering new positions (see the
        portfolio.calculate_nshares() method).
        """
        return sum(self.cash_from_short_positions.values())

    def calculate_nshares(self, entry_price):
        """Returns the number of shares to buy/sell based on free slots, share price and unrestricted cash.

        Example: Portfolio started with initial equity of 10.000 USD and max positions of five. Three long positions
        worth 2000 USD were entered on the same day because there were lots of signals. They all lost 50% of their value
        before closing the trades leaving the portfolio with 7000 USD of unrestricted cash and five free slots.
        New signal comes along the way... a new long position worth 1400 USD (7000 / 5) is entered. Fractional shares
        do not exist here. If the price for one share is 1500 USD, the position will not be entered and the slot will
        remain free. If the price for one share is e.g. 600 USD, a position worth 1200 USD (2 shares) will be entered.

        Another Example: Portfolio started with initial equity of 10.000 USD and max positions of five. One short
        position worth 2000 USD was entered, adding another 2000 USD to the cash balance, which in theory is now
        12.000 USD. Another long signal comes along the way... however, the available cash for opening the new long
        position is 10.000 USD - 2000 USD (restricted cash from the short position) = 8000 USD. This avoids
        over-leveraging, which in most cases will not be part of the intention of the strategies build with the ndxtest
        package.

        Returns
        -------
        number of shares to buy/sell: int
        """
        return (((self.cash - self.restricted_cash()) /
                 (self.max_positions - self.number_of_positions())) / entry_price).__floor__()

    def describe(self, date):
        """Prints a summary of some Portfolio stats to the console."""
        print(f'Stats of {self} as of {date}:')
        print(f'Number of Positions: {self.number_of_positions()}. '
              f'Long: {list(self.long_positions.keys())}, '
              f'Short: {list(self.short_positions.keys())}')
        print(f'Cash: {self.cash.__floor__()}. Cash from shorts: {self.restricted_cash().__floor__()}. '
              f'Invested Capital: {self.invested_capital.__floor__()}.')
        print(f'Total Market Value: {(self.invested_capital + self.cash + self.restricted_cash()).__floor__()}')
        print('\n')

    def create_log_entry(self, entry_data=None, exit_data=None):
        """Creates an entry to self.logdict upon entering or exiting positions.

        Parameters
        ----------
        entry_data: dict, default=None
            Accepts a dict with some data. The dict contains e.g. the entry price and is provided by the backtest class.
        exit_data: dict, default=None
            Accepts a dict with some data. The dict contains e.g. the exit price and is provided by the backtest class.

        Returns
        -------
        A dict with aggregated information about the particular position. The returned dict has the following keys:
        'id', 'entry_date', 'entry_weekday', 'exit_date', 'exit_weekday', 'duration', 'symbol', 'long', 'short',
        'nshares', 'entry_price', 'entry_score', 'entry_value', 'exit_price', 'exit_score', 'exit_value', 'entry_comm',
        'exit_comm', '$change', '%change', 'profitable', 'p.positions', 'p.cash', 'p.res_cash', 'p.invested_capital',
        'market_value'.

        For more information please refer to the online documentation."""
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
        """Creates a pd.DataFrame from self.logdict.

        The DataFrame is used by the instance of the backtest to generate a .xlsx report on the backtest.

        Returns
        -------
        pd.DataFrame
        """
        return pd.DataFrame.from_dict(data=self.logdict, orient='index').round(2)

    def long(self, data):  # closing short position has priority
        """Creates a long, or closes a short position in the portfolio.

        A position is represented by a small dictionary with some information (data). The dictionary becomes a value in
        self.long_positions or self.short_positions with the symbol of the company as the key.

        For more information please refer to the online documentation.
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

            self.logdict[f"{data['exit_date']} {data['symbol']}"] = \
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
                self.logdict[f"{data['entry_date']} {data['symbol']}"] = \
                    self.create_log_entry(entry_data=self.long_positions[data['symbol']])

    def short(self, data):  # closing long position has priority
        """Creates a short, or closes a long position in the portfolio.

        A position is represented by a small dictionary with some information (data). The dictionary becomes a value in
        self.long_positions or self.short_positions with the symbol of the company as the key.

        For more information please refer to the online documentation.
        """
        if data['symbol'] in self.long_positions.keys():
            entry_price = self.long_positions[data['symbol']]['entry_price']
            nshares = self.long_positions[data['symbol']]['nshares']

            self.cash += (nshares * data['exit_price']) - (self.commission * nshares * data['exit_price'])
            self.commission_paid += self.commission * nshares * data['exit_price']
            self.invested_capital -= nshares * (entry_price - data['exit_price'])  # subtracting the P/L
            self.invested_capital -= nshares * data['exit_price']
            self.logdict[f"{data['exit_date']} {data['symbol']}"] = \
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


class BackTest:
    """This is the BackTest class, use it to run backtests of your trading strategies on an index level.

    Attributes
    ----------

    Methods
    -------
    __init__(self, data_path, runtime_messages=True):
        Is responsible for connecting the new instance of the BackTest class to the data folder.
    import_data(self, start_date, end_date, lag, date_range_messages=False):
        Imports the needed price data into pd.Dataframes that are stored in a dictionary.
    generate_signals(self, strategy):
        Generates the trading signals. Accepts an instance of the Strategy class from strategy.py as a parameter.
    run_backtest(...):
        Runs the backtest, please refer to the docstring of the method itself for more information.
    optimize_strategy(self, strategy, parameters, run_best=True):
        Here you can optimize individual parameters of a strategy for better outcomes (Currently not implemented)
    report(self):
        Generates .pdf and .xlsx reports on the backtest in the .../data/output/ directory.
    plot_ticker(self, symbol):
        Plots the price and the signals during the backtest period for a given symbol.
    query_missing_records(self, date):
        Queries, whether any symbols lack price data for a specific date. Helps with debugging the data library.
    setup_search(self, pattern, run_sampling=True):
        Generates a report on how often the strategy generates signals and what price movements happen just after.

    For further information please refer to the docstrings of the respective methods and to the online documentation.
    """

    def __init__(self, data_path, runtime_messages=True):
        # start_date=dt.datetime(2015, 9, 1),
        # end_date=dt.datetime(2021, 9, 1),
        # lag=dt.timedelta(days=200),
        # date_range_messages=False):
        """Defines the class attributes and connects the instance to the data folder. Fails if data folder not present.

        After setting the data_path, this function also performs some tests regarding the contents of the `data` folder
        that are necessary for the proper functioning of the ndxtest package.

        Parameters
        ----------
        data_path: str
            The data_path has to represent the absolute location of the `data` folder.
        """

        if not isinstance(data_path, str):
            raise TypeError('data_path was not of type str')

        if data_path[-1] == '\\':
            pass
        else:
            data_path += '\\'

        if 'data' not in os.listdir(data_path):
            print("Initialization failed.")
            raise FileNotFoundError("data\\ not found in data_path")

        if 'lib' not in os.listdir(data_path + 'data\\'):
            print("Initialization failed.")
            raise FileNotFoundError("data\\ found, but lib\\ folder not found in data\\")

        if '^HIST.xlsx' not in os.listdir(data_path + 'data\\lib\\'):
            print("Initialization failed.")
            raise FileNotFoundError("data\\lib\\ found, but '^HIST.xlsx' is missing.")

        if '^GSPC.csv' not in os.listdir(data_path + 'data\\lib\\'):
            print("Initialization failed.")
            raise FileNotFoundError("data\\lib\\ found, but '^GSPC.csv' is missing.")

        print(f"Setting the DATA_PATH to {data_path} was successful!")

        self.runtime_messages = runtime_messages
        self.data_path = data_path + 'data\\'
        self.output_path = data_path + 'output\\'
        self.data_path_symbols = [symbol[:-4] for symbol in os.listdir(self.data_path + 'lib\\') if '^' not in symbol]
        self.dtypes = {'symbol': str,
                       'open': np.float32,
                       'high': np.float32,
                       'low': np.float32,
                       'close': np.float32,
                       # 'volume': np.int64,
                       'dividends': np.float32,
                       'stock_splits': np.float32}

        self.input_data = {}
        self.data = None
        self.input_index = None
        self.index = None

        self.t0, self.tr, self.runtime = None, None, None

        self.sd = None
        self.ed = None
        self.duration = None
        self.dr = None
        self.edr = None
        self.date_range_messages = None
        self.trading_days = None

        self.constituents = None
        self.existing_symbols = None
        self.missing_symbols = None

        self.commission = None
        self.max_positions = None
        self.initial_equity = None
        self.max_trade_duration = None
        self.stoploss = None
        self.entry_mode = None

        self.signals = defaultdict(list)
        self.eqc = {}
        self.results = {}
        self.drawdown = {}
        self.exposure = {}
        self.correlations = {}
        self.log_df = pd.DataFrame()
        self.equity_curve_df = pd.DataFrame()

        self.opt = False
        self.optimization_results = None
        self.best_parameters = None
        self.parameters = {}
        self.parameter_permutations = []

    def import_data(self, start_date, end_date, lag, date_range_messages=False):

        self.t0 = time.time()
        self.sd, self.ed, self.duration = start_date, end_date, end_date - start_date
        self.dr, self.edr = pd.date_range(start_date, end_date), pd.date_range(start_date - lag, end_date)
        self.date_range_messages = date_range_messages

        self.input_index = pd.read_csv(self.data_path + 'lib\\^GSPC.csv', engine='c', dtype=self.dtypes,
                                       usecols=['date', 'symbol', 'open', 'high', 'low', 'close'],
                                       index_col='date', parse_dates=[0], dayfirst=True)
        self.input_index = self.input_index.loc[self.input_index.index.intersection(self.edr)]

        # self.alerts = "No alerts yet. Use the generate_signals or the scan_for_patterns method first."

        self.constituents = OrderedDict(sorted(constituents(self.sd, self.ed, lag, self.data_path).items()))

        self.existing_symbols = [s for s in self.constituents.keys() if s in self.data_path_symbols
                                 or ('*' in s and s[:-1] in self.data_path_symbols)]

        self.missing_symbols = [s for s in self.constituents.keys() if s not in self.existing_symbols]

        print(f'Importing data...')
        print(f'Constituents in time period: {len(self.constituents.keys())}. '
              f'Thereof not found in data path: {len(self.missing_symbols)}. ')
        print(f'Missing symbols: {self.missing_symbols}')

        for symbol in self.existing_symbols:
            file = f"{self.data_path}lib\\{symbol[:-1] if '*' in symbol else symbol}.csv"
            dr, edr = self.constituents[symbol]['dr'], self.constituents[symbol]['edr']

            with open(file) as f:
                next(f)
                first = next(f).split(',')[0]
                first = dt.datetime.strptime(first, '%Y-%m-%d')
                skip = ((start_date - first).days * 0.65).__floor__() - lag.days

            df = pd.read_csv(file, engine='c', dtype=self.dtypes, skiprows=range(1, skip), index_col='date', parse_dates=['date'])
            df = df.loc[df.index.intersection(edr)]
            df['symbol'] = symbol

            if self.date_range_messages:
                missing_records = df.index.symmetric_difference(edr.intersection(self.input_index.index))
                if any(missing_records):
                    print(f'{symbol}: {len(missing_records)} missing records in extended date range.')

            if not df.empty:
                self.input_data[symbol] = df

        if self.runtime_messages:
            print(f'Data imported:                              ...{(time.time() - self.t0).__round__(2)} sec elapsed.')

    def generate_signals(self, strategy):
        """Write proper docstring!"""
        t1, self.tr = time.time(), time.time()

        strategy.data = self.input_data.copy()
        strategy.index = self.input_index.copy()
        self.data = strategy.generate_signals()
        if self.runtime_messages:
            print(f'Signals generated:                          ...{(time.time() - t1).__round__(2)} sec elapsed.')

        t1 = time.time()

        for symbol, df in self.data.items():
            self.data[symbol] = df.loc[df.index.intersection(self.constituents[symbol]['dr'])]

        self.index = self.input_index.loc[self.input_index.index.intersection(self.dr)]
        self.index['d%change'] = self.index.close.pct_change() * 100
        self.index['c%change'] = ((self.index.close / self.index.close[0]) - 1) * 100
        self.index = self.index.round(2)

        self.trading_days = self.index.index.tolist()

        if self.runtime_messages:
            print(f'Chopping finished:                          ...{(time.time() - t1).__round__(2)} sec elapsed.')
        t1 = time.time()

        concat_data = pd.concat(self.data.values())
        concat_data.drop(columns=['high', 'low', 'close', 'volume', 'dividends', 'stock_splits'], inplace=True)
        dict_data = {}

        self.alerts = concat_data.loc[(concat_data.index == self.trading_days[-1]) &
                                      ((concat_data.entry_signals != 0) | (concat_data.exit_signals != 0))]

        for symbol, df in concat_data.groupby('symbol'):
            df['entry_signals'] = df['entry_signals'].shift(1)  # trades will be executed on the next day
            df['exit_signals'] = df['exit_signals'].shift(1)  # trades will be executed on the next day
            df.loc[df.index[-1], ['entry_signals', 'exit_signals', 'score']] = [0, -2,
                                                                                0]  # adding signals to exit on the last day
            df.dropna(inplace=True)  # dropping NaN values
            dict_data[symbol] = df.loc[(df.entry_signals != 0) | (df.exit_signals != 0)]

        for date, df in pd.concat(dict_data.values()).groupby(level=0):
            self.signals[date] = df.to_dict(orient='records')
            self.signals[date] = sorted(list(self.signals[date]),
                                        key=lambda signal: signal['entry_signals'] * signal['score'], reverse=True)

        if self.runtime_messages:
            print(f'Signals ordered:                            ...{(time.time() - t1).__round__(2)} sec elapsed.')

    def run_backtest(self, long_only=False, commission=.001, max_positions=10, initial_equity=10000.00,
                     max_trade_duration=None, stoploss=None, eqc_method='approx'):
        """Entry Signals: 1 = enter long position, -1 = enter short position,
        Exit Signals: 1 = exit short position, -1 = exit long position, -2 = exit long or short position"""
        self.commission = commission
        self.max_positions = max_positions
        self.initial_equity = initial_equity
        self.max_trade_duration = max_trade_duration
        self.stoploss = stoploss
        self.exposure = {}

        p = Portfolio(max_positions=max_positions, initial_equity=initial_equity, commission=commission)

        t1 = time.time()
        if self.runtime_messages:
            print(f'Running core...')

        for date in self.trading_days:
            max_trade_duration_signals = []
            stoploss_signals = []

            if max_trade_duration is not None and p.positions:
                current_positions = list(p.long_positions.values()) + list(p.short_positions.values())
                max_trade_duration_violated = \
                    list(filter(lambda position: position['entry_date'] <= date - dt.timedelta(days=max_trade_duration),
                                current_positions))
                max_trade_duration_signals = [{'symbol': position['symbol'], 'entry_signals': 0, 'exit_signals': -2,
                                               'score': 0, 'open': self.data[position['symbol']].loc[date, 'open']}
                                              for position in max_trade_duration_violated]

            if stoploss is not None and p.positions:
                long_positions, short_positions = list(p.long_positions.values()), list(p.short_positions.values())
                long_stoploss_violated = list(filter(lambda position:
                                                     position['entry_price'] * (1 - stoploss) >
                                                     self.data[position['symbol']].loc[date, 'open'], long_positions))
                short_stoploss_violated = list(filter(lambda position:
                                                      position['entry_price'] * (1 + stoploss) <
                                                      self.data[position['symbol']].loc[date, 'open'], short_positions))
                stoploss_signals = [{'symbol': position['symbol'], 'entry_signals': 0, 'exit_signals': -2, 'score': 0,
                                     'open': self.data[position['symbol']].loc[date, 'open']}
                                    for position in long_stoploss_violated + short_stoploss_violated]

            self.signals[date] += max_trade_duration_signals
            self.signals[date] += stoploss_signals

            if self.signals[date]:  # signals for this date must exist
                for s in [s for s in self.signals[date] if s['symbol'] in p.positions()]:
                    if s['exit_signals'] in {1, -2} and s['symbol'] in p.short_positions:
                        p.long(data={'symbol': s['symbol'],
                                     'exit_score': s['score'],
                                     'exit_price': s['open'],
                                     'exit_date': date})

                    if s['exit_signals'] in {-1, -2} and s['symbol'] in p.long_positions:
                        p.short(data={'symbol': s['symbol'],
                                      'exit_score': s['score'],
                                      'exit_price': s['open'],
                                      'exit_date': date})

                if date == self.trading_days[-1]:
                    self.signals[date] = []
                else:
                    self.signals[date] = list(filter(lambda signal: signal['symbol'] not in p.positions(),
                                                     self.signals[date]))

                for s in self.signals[date]:
                    if s['entry_signals'] == 1 and s['symbol'] and p.free_slot():
                        s['nshares'] = p.calculate_nshares(s['open'])
                        if s['nshares'] > 0:
                            p.long(data={'symbol': s['symbol'],
                                         'signal': s['entry_signals'],
                                         'entry_score': s['score'],
                                         'entry_price': s['open'],
                                         'nshares': s['nshares'],
                                         'entry_date': date})

                    if not long_only and s['entry_signals'] == -1 and s['symbol'] and p.free_slot():
                        s['nshares'] = -1 * p.calculate_nshares(s['open'])
                        if s['nshares'] < 0:
                            p.short(data={'symbol': s['symbol'],
                                          'signal': s['entry_signals'],
                                          'entry_score': s['score'],
                                          'entry_price': s['open'],
                                          'nshares': s['nshares'],
                                          'entry_date': date})

            # logging the "market exposure" statistic
            self.exposure[date] = p.current_exposure()

            if eqc_method == 'full':
                op, hi, lo, cl, data = 0, 0, 0, 0, None
                if p.positions:
                    for symbol in p.positions():
                        if symbol in p.long_positions.keys():
                            nshares = p.long_positions[symbol]['nshares']
                        else:
                            nshares = p.short_positions[symbol]['nshares']
                        long_positions = self.data[symbol].loc[date, ['open', 'high', 'low', 'close']]
                        op += long_positions.open * nshares
                        hi += long_positions.high * nshares
                        lo += long_positions.low * nshares
                        cl += long_positions.close * nshares
                data = {'open': p.cash + p.restricted_cash() + op, 'high': p.cash + p.restricted_cash() + hi,
                        'low': p.cash + p.restricted_cash() + lo, 'close': p.cash + p.restricted_cash() + cl}
                self.eqc[date] = data

        self.exposure = pd.DataFrame.from_dict(data=self.exposure, orient='index', dtype=np.int8)

        self.log_df = p.create_log_df()
        self.log_df.to_csv(self.output_path + 'tradelog.csv')
        self.log_df = self.log_df.loc[self.log_df['exit_date'] != 0]

        if eqc_method == 'full':
            self.equity_curve_df = pd.DataFrame.from_dict(data=self.eqc, orient='index')
            self.equity_curve_df['d%change'] = self.equity_curve_df.close.pct_change() * 100
            self.equity_curve_df['c%change'] = (((self.equity_curve_df.close / initial_equity) - 1) * 100)
            self.equity_curve_df = self.equity_curve_df.round(2)
            self.equity_curve_df.to_csv(self.output_path + 'equity_curve.csv')

        if eqc_method == 'approx':
            x = self.log_df.drop_duplicates(subset=['exit_date'], keep='last')
            x = x.set_index('exit_date')['market_value']
            self.equity_curve_df = pd.DataFrame(data={'close': np.nan}, index=self.index.index)
            self.equity_curve_df.loc[x.index, 'close'] = x
            self.equity_curve_df.fillna(method="ffill", inplace=True)
            self.equity_curve_df.fillna(self.initial_equity, inplace=True)
            self.equity_curve_df['d%change'] = self.equity_curve_df.close.pct_change() * 100
            self.equity_curve_df['c%change'] = (((self.equity_curve_df.close / initial_equity) - 1) * 100)
            self.equity_curve_df = self.equity_curve_df.round(2)
            self.equity_curve_df.to_csv(self.output_path + 'equity_curve.csv')

        self.drawdown = pd.DataFrame(data={'roll_max': self.equity_curve_df.close.rolling(252, min_periods=1).max()},
                                     index=self.equity_curve_df.index)
        self.drawdown['daily'] = (self.equity_curve_df.close / self.drawdown.roll_max - 1) * 100
        self.drawdown['daily_max'] = self.drawdown.daily.rolling(252, min_periods=1).min()
        self.drawdown = self.drawdown.round(2)

        self.results = {'n_trades': p.number_of_trades,
                        'max_drawdown': self.drawdown.daily_max.min(),
                        'median%pnl': self.log_df['%change'].median().__round__(2),
                        'min%pnl': self.log_df['%change'].min(),
                        'max%pnl': self.log_df['%change'].max(),
                        'median_dur': self.log_df.duration.median(),
                        'min_dur': self.log_df.duration.min(),
                        'max_dur': self.log_df.duration.max(),
                        '%profitable': (np.count_nonzero(self.log_df.profitable != 0) / len(
                            self.log_df.profitable) * 100).__round__(2),
                        'commission_paid': p.commission_paid.__round__(2),
                        'perf': self.equity_curve_df['c%change'][-1],
                        'ann_perf': (((1 + ((self.equity_curve_df.close[-1] - initial_equity) / initial_equity))
                                      ** (1 / (self.duration.days / 365.25)) - 1) * 100).__round__(2),
                        'bm_perf': self.index['c%change'][-1],
                        'bm_ann_perf': (((1 + (
                                (self.index.close[-1] - self.index.close[0]) / self.index.close[0]))
                                         ** (1 / (self.duration.days / 365.25)) - 1) * 100).__round__(2)}

        if self.runtime_messages:
            print(f'Finished:                                   ...{(time.time() - t1).__round__(2)} sec elapsed.')
            print(
                f'Time per trading day:                       ...{((time.time() - t1) / len(self.trading_days) * 1000).__round__(2)} ms.')

        self.runtime = (time.time() - self.tr).__round__(2)
        if self.runtime_messages:
            print(f'Total runtime ex. data import:              ...{self.runtime} sec.')

    def optimize_strategy(self, strategy, parameters, run_best=True):
        # raise NotImplementedError('This class method is currently not implemented!')

        def frange(x, y, jump):
            while x <= y:
                yield float(x)
                x += decimal.Decimal(jump)

        self.parameters = parameters

        for k in self.parameters.keys():
            if not parameters[k]:
                continue
            if any(isinstance(v, float) for v in parameters[k]):
                lo, hi, step = tuple(decimal.Decimal(str(x)) for x in parameters[k])
                self.parameters[k] = [p for p in frange(lo, hi, step)]
            else:
                lo, hi, step = parameters[k]
                self.parameters[k] = [p for p in range(lo, hi + step, step)]

        print(self.parameters)

        permutations = [element for element in itertools.product(*self.parameters.values())]
        self.parameter_permutations = [{k: v for (k, v) in zip(self.parameters.keys(), comb)} for comb in permutations]

        print(f'There are {len(self.parameter_permutations)} parameter combinations to run.')
        print(
            f'Running the optimization will approximately take {(len(self.parameter_permutations) * self.runtime * 1.2).__round__(2)} seconds.')
        if input(f'Proceed (Y/N): ').upper() == 'Y':
            self.runtime_messages = False
            s = []
            d = pd.DataFrame(columns=list(self.parameters.keys()))
            for parameters in self.parameter_permutations:
                self.generate_signals(strategy, parameters)
                self.run_backtest(long_only=True,
                                  commission=self.commission,
                                  max_positions=self.max_positions,
                                  initial_equity=self.initial_equity,
                                  max_trade_duration=self.max_trade_duration,
                                  stoploss=self.stoploss,
                                  eqc_method='approx')

                print(f'{parameters} --> {self.results["perf"]}')

                s.append(self.results['perf'])
                d = d.append(parameters, ignore_index=True)

            self.optimization_results = pd.Series(data=s, index=pd.MultiIndex.from_frame(d), name='performance')
            self.optimization_results.to_csv('output\\optimization_results.csv')

            combs = list(itertools.combinations(list(self.parameters.keys()), 2))

            fig3, axs3 = plt.subplots(1, len(combs), figsize=(9, 9 / len(combs)), gridspec_kw={"wspace": .5})
            for i, combination in enumerate(combs):
                hm = self.optimization_results.groupby(list(combination)).mean().unstack()
                sns.heatmap(hm[::], annot=True, ax=axs3 if len(combs) == 1 else axs3[i], cmap='viridis')
                mp = [p for p in list(self.parameters.keys()) if p not in combination]
                axs3.title.set_text(f'Performance') if len(combs) == 1 else axs3[i].title.set_text(
                    f'Mean of all {mp} runs.')

            plt.savefig("f3.png", dpi=None, facecolor='w', edgecolor='w')

            self.best_parameters = {f'p{i + 1}': val for i, val in enumerate(self.optimization_results.idxmax())}
            self.opt = True

            if run_best:
                self.generate_signals(strategy, self.best_parameters)
                self.run_backtest(long_only=True,
                                  commission=self.commission,
                                  max_positions=self.max_positions,
                                  initial_equity=self.initial_equity,
                                  max_trade_duration=self.max_trade_duration,
                                  stoploss=self.stoploss,
                                  eqc_method='full')

    def report(self):

        fig1, axs1 = plt.subplots(3, 1, figsize=(9, 6))
        axs1[0].plot(self.index['c%change'].index, self.index['c%change'], color='blue', label='Benchmark')
        axs1[0].plot(self.equity_curve_df['c%change'].index, self.equity_curve_df['c%change'], color='black',
                     label='Backtest')
        axs1[0].set_xticks([])
        axs1[0].set_xticklabels([])
        axs1[0].set_ylabel('Cumulative % Change')
        axs1[0].grid(axis='y')
        axs1[0].legend()
        axs1[1].plot(self.drawdown['daily'].index, self.drawdown['daily'], color='black', label='Drawdown')
        axs1[1].plot(self.drawdown['daily_max'].index, self.drawdown['daily_max'], color='red',
                     label='Max. 1Yr Drawdown')
        axs1[1].set_xticks([])
        axs1[1].set_xticklabels([])
        axs1[1].set_ylabel('1Yr Drawdown')
        axs1[1].grid(axis='y')
        axs1[1].legend()
        axs1[2].plot(self.exposure['n_long'].index, self.exposure['n_long'], color='blue', label='Long Positions')
        axs1[2].plot(self.exposure['n_short'].index, self.exposure['n_short'], color='red', label='Short Positions')
        axs1[2].plot(self.exposure['n_free'].index, self.exposure['n_free'], color='grey', label='Free Slots')
        axs1[2].set_xlabel('Date')
        axs1[2].set_ylabel('Positions')
        axs1[2].grid(axis='y')
        axs1[2].legend()
        fig1.tight_layout()

        plt.savefig(self.output_path + "f1.png", dpi=None, facecolor='w', edgecolor='w')

        fig2, axs2 = plt.subplots(1, 3, figsize=(9, 3), gridspec_kw={"wspace": .5})
        main_corr = self.log_df[['entry_score', 'duration', 'entry_price', 'profitable']]
        self.correlations['main_corr'] = pd.DataFrame(data=main_corr.corr()['profitable'],
                                                      index=['entry_score', 'duration', 'entry_price', 'profitable'])
        axs2[0].title.set_text('General Correlations')
        sns.heatmap(self.correlations['main_corr'], annot=True, square=False, ax=axs2[0], cmap='viridis')

        entry_corr = pd.DataFrame()
        entry_corr['mon'] = self.log_df['entry_weekday'] == 1
        entry_corr['tue'] = self.log_df['entry_weekday'] == 2
        entry_corr['wed'] = self.log_df['entry_weekday'] == 3
        entry_corr['thu'] = self.log_df['entry_weekday'] == 4
        entry_corr['fri'] = self.log_df['entry_weekday'] == 5
        entry_corr['profitable'] = self.log_df['profitable']
        self.correlations['entry_corr'] = pd.DataFrame(data=entry_corr.corr()['profitable'],
                                                       index=['mon', 'tue', 'wed', 'thu', 'fri', 'profitable'])
        axs2[1].title.set_text('Entry Weekday Correlations')
        sns.heatmap(self.correlations['entry_corr'], annot=True, square=False, ax=axs2[1], cmap='viridis')

        exit_corr = pd.DataFrame()
        exit_corr['mon'] = self.log_df['exit_weekday'] == 1
        exit_corr['tue'] = self.log_df['exit_weekday'] == 2
        exit_corr['wed'] = self.log_df['exit_weekday'] == 3
        exit_corr['thu'] = self.log_df['exit_weekday'] == 4
        exit_corr['fri'] = self.log_df['exit_weekday'] == 5
        exit_corr['profitable'] = self.log_df['profitable']
        self.correlations['exit_corr'] = pd.DataFrame(data=exit_corr.corr()['profitable'],
                                                      index=['mon', 'tue', 'wed', 'thu', 'fri', 'profitable'])
        axs2[2].title.set_text('Exit Weekday Correlations')
        sns.heatmap(self.correlations['exit_corr'], annot=True, square=False, ax=axs2[2], cmap='viridis')

        plt.savefig(self.output_path + "f2.png", dpi=None, facecolor='w', edgecolor='w')

        class PDF(FPDF):  # A4: w = 210, h = 297
            pass

        pdf = PDF()
        pdf.set_left_margin(margin=30)
        pdf.add_page()
        pdf.set_font('helvetica', size=12)

        pdf.cell(w=80, align='', txt=f'BackTest Report:', ln=0)
        pdf.cell(w=80, align='', txt=f'{dt.datetime.now().date()}', ln=1)
        pdf.cell(w=80, align='', txt=f'Benchmark Index:', ln=0)
        pdf.cell(w=80, align='', txt=f'SPY', ln=1)

        pdf.ln(140)
        pdf.image(self.output_path + 'f1.png', x=15, y=25, w=180, h=120)

        pdf.cell(w=80, align='', txt=f'Parameters:', ln=0)
        pdf.cell(w=80, align='', txt=f'{self.best_parameters if self.opt else "not optimized"}', ln=1)
        pdf.cell(w=80, align='', txt=f'Start date:', ln=0)
        pdf.cell(w=80, align='', txt=f'{self.sd.date()}', ln=1)
        pdf.cell(w=80, align='', txt=f'End date:', ln=0)
        pdf.cell(w=80, align='', txt=f'{self.ed.date()}', ln=1)
        pdf.cell(w=80, align='', txt=f"Initial Equity [$]:", ln=0)
        pdf.cell(w=80, align='', txt=f"{self.initial_equity}", ln=1)
        pdf.cell(w=80, align='', txt=f"Final Equity [$]:", ln=0)
        pdf.cell(w=80, align='', txt=f"{self.equity_curve_df.close[-1]}", ln=1)
        pdf.cell(w=80, align='', txt=f"Max. Drawdown [%]:", ln=0)
        pdf.cell(w=80, align='', txt=f"{self.results['max_drawdown']}", ln=1)
        pdf.cell(w=80, align='', txt=f"Performance Strategy [%]:", ln=0)
        pdf.cell(w=80, align='', txt=f"{self.results['perf']}", ln=1)
        pdf.cell(w=80, align='', txt=f"Performance Benchmark [%]:", ln=0)
        pdf.cell(w=80, align='', txt=f"{self.results['bm_perf']}", ln=1)
        pdf.cell(w=80, align='', txt=f"Ann. Performance Strategy [%]:", ln=0)
        pdf.cell(w=80, align='', txt=f"{self.results['ann_perf']}", ln=1)
        pdf.cell(w=80, align='', txt=f"Ann. Performance Benchmark [%]:", ln=0)
        pdf.cell(w=80, align='', txt=f"{self.results['bm_ann_perf']}", ln=1)
        pdf.cell(w=80, align='', txt=f"Total commissions paid [$]:", ln=0)
        pdf.cell(w=80, align='', txt=f"{self.results['commission_paid']}", ln=1)
        pdf.cell(w=80, align='', txt=f"Number of trades executed:", ln=0)
        pdf.cell(w=80, align='', txt=f"{self.results['n_trades']}", ln=1)
        pdf.cell(w=80, align='', txt=f"Median trade duration [Days]:", ln=0)
        pdf.cell(w=80, align='', txt=f"{self.results['median_dur']}", ln=1)
        pdf.cell(w=80, align='', txt=f"Worst trade [%]:", ln=0)
        pdf.cell(w=80, align='', txt=f"{self.results['min%pnl']}", ln=1)
        pdf.cell(w=80, align='', txt=f"Best trade [%]:", ln=0)
        pdf.cell(w=80, align='', txt=f"{self.results['max%pnl']}", ln=1)
        pdf.cell(w=80, align='', txt=f"Median trade [%]:", ln=0)
        pdf.cell(w=80, align='', txt=f"{self.results['median%pnl']}", ln=1)
        pdf.cell(w=80, align='', txt=f"Profitable trades [%]:", ln=0)
        pdf.cell(w=80, align='', txt=f"{self.results['%profitable']}", ln=1)

        pdf.add_page()
        # pdf.ln(10)
        # pdf.cell(w=190, align='C', txt=f"Correlations with profitability of trades:")
        # pdf.image('output\\f2.png', x=15, y=20, w=180, h=60)
        if self.opt:
            pdf.cell(w=190, align='', txt=f"Optimization:")
            shape = img.imread(self.output_path + 'f3.png').shape
            pdf.image(self.output_path + 'f3.png', x=10, y=20, w=shape[0] * (160 / shape[0]),
                      h=shape[1] * (160 / shape[1]))

        pdf.output(self.output_path + 'test.pdf', 'F')

        os.remove(self.output_path + 'f1.png')
        os.remove(self.output_path + 'f2.png')
        if self.opt:
            os.remove(self.output_path + 'f3.png')

    def plot_ticker(self, symbol):
        """This function will plot."""

        df = self.data[symbol]

        long_entries = df['entry_signals'].replace([0, -1], np.nan)
        long_exits = df['exit_signals'].replace([0, 1], np.nan)
        short_entries = df['entry_signals'].replace([0, 1], np.nan)
        short_exits = df['exit_signals'].replace([0, -1], np.nan)

        long_entries = mpf.make_addplot(long_entries, color='green', panel=2, ylim=(-3, 3), secondary_y=False,
                                        type="scatter", markersize=20, marker='^', ylabel='Long Signals')
        long_exits = mpf.make_addplot(long_exits, color='red', panel=2, ylim=(-3, 3), secondary_y=False,
                                      type="scatter", markersize=20, marker='v')
        short_entries = mpf.make_addplot(short_entries, color='green', panel=3, ylim=(-3, 3), secondary_y=False,
                                         type="scatter", markersize=20, marker='^', ylabel='Short Signals')
        short_exits = mpf.make_addplot(short_exits, color='red', panel=3, ylim=(-3, 3), secondary_y=False,
                                       type="scatter", markersize=20, marker='v')

        addplts = [long_entries, long_exits, short_entries, short_exits]

        mpf.plot(self.data[symbol], type="candle", style='blueskies', xrotation=45, volume=True,
                 addplot=addplts, panel_ratios=(1, 0.5, 0.5, 0.5), warn_too_much_data=10000)

    def query_missing_records(self, date):
        """(for debugging) Queries, whether any symbols lack price data for a specific date.

        Parameters
        ----------
        date: datetime.datetime object or str
            If string it must be 'YYYY-MM-DD' formatted.

        Returns
        -------
        None. It prints the results in the console.
        """

        if not isinstance(date, dt.datetime):
            if isinstance(date, str):
                date = dt.datetime.strptime(date, '%Y-%m-%d')
            else:
                raise TypeError("Please provide a 'YYYY-MM-DD' formatted string or a datetime.datetime object.")

        if date.isoweekday() in {6, 7}:
            raise ValueError(f'{date} is not a weekday. Call function with weekdays only.')

        if date < self.sd:
            print(f'Warning: {date} is not within the backtest period. The record might be available but not imported.')

        for k, v in self.input_data.items():
            if date not in v.index:
                if date in self.constituents[k]['dr']:
                    print(f'Data for {k} is missing the record for {date}.')

    def setup_search(self, pattern, run_sampling=True):
        """This function searches the input_data for certain patterns that represent buy or sell signals.
        The pattern parameter is a list of conditions that have to be met in order to generate a signal in the
        following general format:
        [(-n, cond, False), ..., (-1, cond, False), (0, cond, False), (1, cond, True)]
        Wherein: 0 represents the day of signal completion, 1 represents the day of signal execution
        (buy/sell transaction), -1 represents one time period (day) prior to the signal completion and so on.
        cond represent conditions that have to be met. 'False' means that the condition has to be met in the
        respective ticker symbol. 'True' means that the condition has to be met in the ^GSPC (S&P 500) index.
        As an example, the pattern:
        [(-1, bullish_pin_bar(), False), (0, gap_up(), False), (1, gap_up(), True)]
        Translates into: On day -1 a bullish pin bar has to form. On the day of signal completion the ticker symbol
        has to gap up. On the day of transaction the index has to gap up. all(pattern) has to be True otherwise a
        transaction is not triggered.
        Patterns should (but don't have to) be provided in chronological order. One day can have several conditions
        (separate tuple for each condition)
        """

        if isinstance(pattern, list):

            results, random_results = {}, {}

            # scanning for input pattern in symbol dfs
            for symbol, df in self.input_data.items():

                signals = pd.DataFrame()

                for i, element in enumerate(pattern):
                    delta_days, func, index = element  # how will funcs with arguments be handled?
                    if not index:
                        # task find more elegant way of column naming?
                        signals[i] = func(df).shift(-delta_days)
                    else:
                        signals[i] = func(self.input_index).shift(-delta_days)

                cds = signals.loc[signals.T.all()].index.values  # cds = completion days of pattern (=0)

                results[symbol] = pd.DataFrame(data={'cd': cds,
                                                     'symbol': symbol,
                                                     'ep': df.shift(-1).loc[cds, 'open'],  # entry price
                                                     'c': df.shift(-1).loc[cds, 'close'],
                                                     'c_gt_ep': df.shift(-1).loc[cds, 'open'] < df.shift(-1).loc[
                                                         cds, 'close'],
                                                     '+1o': df.shift(-2).loc[cds, 'open'],
                                                     '+1o_gt_ep': df.shift(-1).loc[cds, 'open'] < df.shift(-2).loc[
                                                         cds, 'open'],
                                                     '+1c': df.shift(-2).loc[cds, 'close'],
                                                     '+1c_gt_ep': df.shift(-1).loc[cds, 'open'] < df.shift(-2).loc[
                                                         cds, 'close'],
                                                     '+2o': df.shift(-3).loc[cds, 'open'],
                                                     '+2o_gt_ep': df.shift(-1).loc[cds, 'open'] < df.shift(-3).loc[
                                                         cds, 'open'],
                                                     '+2c': df.shift(-3).loc[cds, 'close'],
                                                     '+2c_gt_ep': df.shift(-1).loc[cds, 'open'] < df.shift(-3).loc[
                                                         cds, 'close']
                                                     })

            d = pd.concat(results, ignore_index=True).sort_values(by='cd')

            if len(d.index) == 0:
                print('No instances of pattern found!')
            else:
                d.dropna(inplace=True)

                # now generating random results
                for i, symbol in enumerate(random.choices(list(self.input_data.keys()), k=1000)):
                    df = self.input_data[symbol]
                    cd = random.choice(df.index.values)
                    random_results[i] = {'cd': cd,
                                         'symbol': symbol,
                                         'ep': df.shift(-1).loc[cd, 'open'],  # entry price
                                         'c': df.shift(-1).loc[cd, 'close'],
                                         'c_gt_ep': df.shift(-1).loc[cd, 'open'] < df.shift(-1).loc[
                                             cd, 'close'],
                                         '+1o': df.shift(-2).loc[cd, 'open'],
                                         '+1o_gt_ep': df.shift(-1).loc[cd, 'open'] < df.shift(-2).loc[
                                             cd, 'open'],
                                         '+1c': df.shift(-2).loc[cd, 'close'],
                                         '+1c_gt_ep': df.shift(-1).loc[cd, 'open'] < df.shift(-2).loc[
                                             cd, 'close'],
                                         '+2o': df.shift(-3).loc[cd, 'open'],
                                         '+2o_gt_ep': df.shift(-1).loc[cd, 'open'] < df.shift(-3).loc[
                                             cd, 'open'],
                                         '+2c': df.shift(-3).loc[cd, 'close'],
                                         '+2c_gt_ep': df.shift(-1).loc[cd, 'open'] < df.shift(-3).loc[
                                             cd, 'close']
                                         }
                rnd = pd.DataFrame.from_dict(random_results, orient='index').sort_values(by='cd')
                rnd.dropna(inplace=True)

                cols = [('c', 'c_gt_ep'), ('+1o', '+1o_gt_ep'), ('+1c', '+1c_gt_ep'), ('+2o', '+2o_gt_ep'),
                        ('+2c', '+2c_gt_ep')]
                stats = {'metric': ['Total count:',
                                    'T count:',
                                    'F count:',
                                    'T %:',
                                    'F %:',
                                    'Avg. % change:',
                                    'Med. % change:',
                                    'Std. %:',
                                    'T avg. % change:',
                                    'T med. % change:',
                                    'T std. %:',
                                    'T % change min:',
                                    'T % change max:',
                                    'F avg. % change:',
                                    'F med. % change:',
                                    'F std. %:',
                                    'F % change min:',
                                    'F % change max:',
                                    ]}

                # making stats from the experimental data
                for price_col, count_col in cols:
                    fc, tc = d[count_col].value_counts().sort_index().tolist()
                    pct_change = ((d[price_col] / d['ep']) - 1) * 100
                    t_pct_change = ((d.loc[d[count_col], price_col] / d.loc[d[count_col], 'ep']) - 1) * 100
                    f_pct_change = ((d.loc[~d[count_col], price_col] / d.loc[~d[count_col], 'ep']) - 1) * 100
                    stats[count_col] = [tc + fc,
                                        tc,
                                        fc,
                                        round((tc / (fc + tc) * 100), 2),
                                        round((fc / (fc + tc) * 100), 2),
                                        round(pct_change.mean(), 2),
                                        round(pct_change.median(), 2),
                                        round(pct_change.std(ddof=1), 2),
                                        round(t_pct_change.mean(), 2),
                                        round(t_pct_change.median(), 2),
                                        round(t_pct_change.std(ddof=1), 2),
                                        round(t_pct_change.min(), 2),
                                        round(t_pct_change.max(), 2),
                                        round(f_pct_change.mean(), 2),
                                        round(f_pct_change.median(), 2),
                                        round(f_pct_change.std(ddof=1), 2),
                                        round(f_pct_change.min(), 2),
                                        round(f_pct_change.max(), 2)
                                        ]

                # making stats from random data
                stats[' '] = ''  # a spaceholder
                for price_col, count_col in cols:
                    fc, tc = rnd[count_col].value_counts().sort_index().tolist()
                    pct_change = ((rnd[price_col] / rnd['ep']) - 1) * 100
                    t_pct_change = ((rnd.loc[rnd[count_col], price_col] / rnd.loc[rnd[count_col], 'ep']) - 1) * 100
                    f_pct_change = ((rnd.loc[~rnd[count_col], price_col] / rnd.loc[~rnd[count_col], 'ep']) - 1) * 100
                    stats[count_col + '_rnd'] = [tc + fc,
                                                 tc,
                                                 fc,
                                                 round((tc / (fc + tc) * 100), 2),
                                                 round((fc / (fc + tc) * 100), 2),
                                                 round(pct_change.mean(), 2),
                                                 round(pct_change.median(), 2),
                                                 round(pct_change.std(ddof=1), 2),
                                                 round(t_pct_change.mean(), 2),
                                                 round(t_pct_change.median(), 2),
                                                 round(t_pct_change.std(ddof=1), 2),
                                                 round(t_pct_change.min(), 2),
                                                 round(t_pct_change.max(), 2),
                                                 round(f_pct_change.mean(), 2),
                                                 round(f_pct_change.median(), 2),
                                                 round(f_pct_change.std(ddof=1), 2),
                                                 round(f_pct_change.min(), 2),
                                                 round(f_pct_change.max(), 2)
                                                 ]

                sampling = {'metric': ['Dist type:',
                                       'Dist mean:',
                                       'Dist std:',
                                       'Std ddof:',
                                       'Sample size:',
                                       'Repetitions:',
                                       'Init. capital:',
                                       'Commission:',
                                       'Result min:',
                                       'Result max:',
                                       'Result mean:',
                                       'Result median:',
                                       'Result std:',
                                       'P<.05 vs rnd:',
                                       'P<.01 vs rnd:']}
                if run_sampling:
                    # making stats from random data

                    for price_col, count_col in cols:
                        dist_m, dist_std = stats[count_col][5], stats[count_col][7]
                        ddof, ssize, reps, init_cap, comm = 1, 100, 100, 10000, 0.001
                        sampling_results = []
                        for i in range(0, reps):
                            cap = init_cap
                            for n in np.random.normal(dist_m, dist_std, ssize):
                                cap = cap - (cap * comm * 2)
                                cap = cap * (1 + n / 100)
                            sampling_results.append(cap)
                        sampling_results = pd.Series(data=sampling_results)

                        sampling[count_col] = ['normal',
                                               dist_m,
                                               dist_std,
                                               ddof,
                                               ssize,
                                               reps,
                                               init_cap,
                                               comm,
                                               sampling_results.min(),
                                               sampling_results.max(),
                                               sampling_results.mean(),
                                               sampling_results.median(),
                                               sampling_results.std(ddof=ddof),
                                               0,
                                               0]

                with pd.ExcelWriter('pattern_stats.xlsx') as writer:
                    d.to_excel(writer, sheet_name='pattern', index=False)
                    rnd.to_excel(writer, sheet_name='random', index=False)
                    pd.DataFrame(data=stats).to_excel(writer, sheet_name='stats_pattern_vs_random', index=False)
                    pd.DataFrame(data=sampling).to_excel(writer, sheet_name='sampling_results', index=False)
