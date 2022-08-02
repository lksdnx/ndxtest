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

import os
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import pandas as pd
import decimal
import itertools
from collections import OrderedDict, defaultdict
from ndxtest.utils import Portfolio, constituents
import time
import datetime as dt
import mplfinance as mpf
import random
import seaborn as sns
from fpdf import FPDF


class BackTest:
    """This is the BackTest class, use it to run backtests of your trading strategies on an index level.

    This doctsring primarily focuses on listing the class attributes and class methods. For information on how to use
    this class please refer to the online documentation.

    The 'default' values refer to the values set during initialization of an instance of the class.

    Attributes
    ----------
    self.runtime_messages: bool, default=True
        If True, prints elapsed time for data import and other computations to the console.
    self.data_path: str
        The path to the data folder.
    self.output_path: str, default=self.data_path + 'data\\output\\'
        The path to the output folder.
    self.data_path_symbols: list
        List of all symbols for which .csv files were found in data\\lib.
    self.dtypes: dict
        Dictionary containing the dtypes to use for columns during import of data.

    self.input_data: dict, default={}
        Dictionary containing the imported price data. {'AAPL': pd.DataFrame(...), ..., 'ZTS': pd.DataFrame(...)}
    self.data: dict, default=None
        Copy of self.input_data with computed trading signals added to the pd.DataFrames.
    self.input_index: pd.DataFrame, default=None
        pd.DataFrame containing the price data of the reference index. As of now the S&P 500 (Ticker Symbol = '^GSPC')
    self.index: pd.DataFrame, default=None
        Copy of self.index with computed indicators (needed for signal generation) added to the DataFrame.

    self.t0, default=None
        Used internally to store times needed for computations.
    self.tr, default=None
        Used internally to store times needed for computations.
    self.runtime, default=None
        Used internally to store times needed for computations.

    self.sd: datetime.datetime object or str, default=None
        The start date of the backtest. If weekend or US market closed, the next trading day will be set as start date.
    self.ed: datetime.datetime object or str, default=None
        The end date of the backtest. If weekend or US market closed, the next trading day will be set as end date.
    self.duration: datetime.timedelta object, default=None
        The duration of the backtest.
    self.dr: pd.DateRange, default=None
        The range of dates between start and end date of the backtest.
    self.edr: pd.DateRange, default=None
        The range of dates added in front due to the lag parameter, plus the range of dates between start and end date.
    self.date_range_messages: bool, default=False
        If True, prints some extra information regarding missing price data for the backtest period to the console.
    self.trading_days: list, default=None
        A list of all trading days between start and end date. (Intersection of self.dr and self.input_index.index)

    self.constituents: dict, default=None
        A nested dictionary. See ndxtest.utils.constituents() docstring.
    self.existing_symbols: list, default=None
        A list of all symbols included in the index during the specified time period and exist in data\\lib.
    self.missing_symbols: list, default=None
        A list of all symbols included in the index during the specified time period and DO NOT exist in data\\lib.

    self.commission: float, default=None
        Commission that is paid upon entering/exiting positions (e.g. 0.01 would be 1%)
    self.max_positions: int, default=None
        Maximum number of positions (slots). No positions are entered when all slots are filled.
    self.initial_equity: float, default=None
        The initial capital to start with.
    self.max_trade_duration: int, default=None
        A maximum number of days before positions will be (independent of signals) closed.
    self.stoploss: float, default=None
        The maximum % (e.g. 0.05) of adverse price movement before positions will be (independent of signals) closed.
    self.entry_mode: str, default=None
        The mode of entry. 'open' = buy on open to the day after signal completion. Currently only this is implemented.

    self.signals: defaultdict(list), default=defaultdict(list)
        A dictionary containing the computed signals with the respective trading days as keys.
    self.eqc: dict, default={}
        Is filled with daily ohlc date for the portfolio while the backtest runs. Used to generate the equity curve.
    self.eqc_df: pd.DataFrame(), default=pd.DataFrame()
        A pd.DataFrame generated from self.eqc is stored in this attribute.
    self.results: dict, default={}
        Is filled with some high-level results (e.g. the winrate) of the backtest. Used to generate reports.
    self.drawdown: dict, default={}
        Is filled with data used to calculate the (yearly) maximum drawdown.
    self.exposure: dict, default={}
        Is filled with info on the exposure to the market over time (e.g. invested cap vs cash, number of positions)
    self.correlations: dict, default={}
        -- currently to implemented!
    self.log_df: pd.DataFrame(), default=pd.DataFrame()
        Is keeping a log of trades taken. Used to generate reports.

    self.opt: bool, default=False
        ...currently not implemented!
    self.optimization_results: pd.Series, default=None
        ...currently not implemented!
    self.best_parameters: dict, default=None
        ...currently not implemented!
    self.parameters: dict, default={}
        ...currently not implemented!
    self.parameter_permutations: list, default=[]
        ...currently not implemented!


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
        """Defines the class attributes and connects the instance to the data folder. Fails if data folder not present.

        After setting the data_path, this function also performs some tests regarding the contents of the `data` folder
        that are necessary for the proper functioning of the ndxtest package.

        :param str data_path:
            The data_path has to represent the absolute location of the `data` folder.

        :returns:
            None
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

        self.t0 = None
        self.tr = None
        self.runtime = None

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
        self.eqc_df = pd.DataFrame()
        self.results = {}
        self.drawdown = {}
        self.exposure = {}
        self.correlations = {}
        self.log_df = pd.DataFrame()

        self.opt = False
        self.optimization_results = None
        self.best_parameters = None
        self.parameters = {}
        self.parameter_permutations = []

    def import_data(self, start_date, end_date, lag, date_range_messages=False):
        """Imports the necessary price data for the defined time period of the backtest.

        :param datetime.datetime or str start_date:
            The start date of the backtest. If weekend or US market closed, the next trading day will be set as start date.
        :param datetime.datetime or str end_date:
            The end date of the backtest. If weekend or US market closed, the next trading day will be set as end date.
        :param datetime.timedelta or int lag:
            A timedelta that is added in front of the start_date. This is necessary for calculating indicators that
            depend on previous price data such as moving averages among others.
        :param bool date_range_messages:
            If True, prints some extra information regarding missing price data for the backtest period to the console.

        :returns: None
        """

        if not isinstance(start_date, dt.datetime):
            if not isinstance(lag, str):
                raise TypeError("Parameter start_date must be of type datetime.datetime or str formatted 'YYYY-MM-DD'.")
            else:
                self.sd = dt.datetime.strptime(date_string=start_date, format='%Y-%m-%d')
        else:
            self.sd = start_date

        if not isinstance(end_date, dt.datetime):
            if not isinstance(lag, str):
                raise TypeError("Parameter start_date must be of type datetime.datetime or str formatted 'YYYY-MM-DD'.")
            else:
                self.ed = dt.datetime.strptime(date_string=start_date, format='%Y-%m-%d')
        else:
            self.sd = end_date

        if not isinstance(lag, dt.timedelta):
            if not isinstance(lag, int):
                raise TypeError('Parameter lag must be of type datetime.timedelta or int.')
            else:
                lag = dt.timedelta(days=lag)

        self.t0 = time.time()
        self.duration = self.ed - self.sd
        self.dr = pd.date_range(self.sd, self.ed)
        self.edr = pd.date_range(self.sd - lag, self.ed)
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
            self.eqc_df = pd.DataFrame.from_dict(data=self.eqc, orient='index')
            self.eqc_df['d%change'] = self.eqc_df.close.pct_change() * 100
            self.eqc_df['c%change'] = (((self.eqc_df.close / initial_equity) - 1) * 100)
            self.eqc_df = self.eqc_df.round(2)
            self.eqc_df.to_csv(self.output_path + 'equity_curve.csv')

        if eqc_method == 'approx':
            x = self.log_df.drop_duplicates(subset=['exit_date'], keep='last')
            x = x.set_index('exit_date')['market_value']
            self.eqc_df = pd.DataFrame(data={'close': np.nan}, index=self.index.index)
            self.eqc_df.loc[x.index, 'close'] = x
            self.eqc_df.fillna(method="ffill", inplace=True)
            self.eqc_df.fillna(self.initial_equity, inplace=True)
            self.eqc_df['d%change'] = self.eqc_df.close.pct_change() * 100
            self.eqc_df['c%change'] = (((self.eqc_df.close / initial_equity) - 1) * 100)
            self.eqc_df = self.eqc_df.round(2)
            self.eqc_df.to_csv(self.output_path + 'equity_curve.csv')

        self.drawdown = pd.DataFrame(data={'roll_max': self.eqc_df.close.rolling(252, min_periods=1).max()},
                                     index=self.eqc_df.index)
        self.drawdown['daily'] = (self.eqc_df.close / self.drawdown.roll_max - 1) * 100
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
                        'perf': self.eqc_df['c%change'][-1],
                        'ann_perf': (((1 + ((self.eqc_df.close[-1] - initial_equity) / initial_equity))
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
        axs1[0].plot(self.eqc_df['c%change'].index, self.eqc_df['c%change'], color='black',
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
        pdf.cell(w=80, align='', txt=f"{self.eqc_df.close[-1]}", ln=1)
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
