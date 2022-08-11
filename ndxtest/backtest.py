"""This is the main module of the ndxtest package containing :class:`ndxtest.backtest.BackTest` and
:class:`ndxtest.backtest.Strategy`.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import pandas as pd
import decimal
import itertools
from collections import OrderedDict, defaultdict
from ndxtest.utils import Portfolio, constituents, connect, timeit_decorator
import time
import datetime as dt
import mplfinance as mpf
import random
import seaborn as sns
from fpdf import FPDF


class Strategy:
    """Class to build strategies. Strategy objects are provided as a parameter to the `generate_signals` method of
    :class:`ndxtest.backtest.BackTest`.

    :ivar dict data: A dictionary containing price data provided by :class:`ndxtest.backtest.BackTest`. Initial value: None
    :ivar pd.Dataframe index: A a pd.Dataframe containing price data of the reference index. Initial value: None
    :ivar dict condition_sets: A dict of lists with conditions for entering and exiting positions. Initial value: {...}

    For information on how to use this class please refer to the user manual.
    """

    def __init__(self, data=None, index=None):
        """Constructor Method.

        :param dict data: The is data provided when the strategy is provided to the `generate_signals` method of :class:`ndxtest.backtest.BackTest`. Initial value: None
        :param pd.DataFrame index: Is provided when the strategy is provided to the `generate_signals` method of :class:`ndxtest.backtest.BackTest`. Initial value: None
        """

        self.data = data
        self.index = index
        # conditions for entering (en) and exiting (ex) long (l) and short (s) trades
        self.condition_sets = {'enl': [],
                               'exl': [],
                               'ens': [],
                               'exs': []}

    def enter_long_if(self, day, condition, use_index=False):
        """Adding an entry condition for long positions.

        :param int day: Day (relative to day 0 of signal completion) on which to check for the condition to be fulfilled.
        :param function condition: A function of x where x is a `pd.DataFrame` that returns a boolean `pd.Series` with the same index.
        :param bool, default=False use_index: If True, is being applied to the symbol df, else to the reference index df.

        For information on how conditions can be structured please refer to the user manual.

        :returns: None
        :rtype: NoneType
        """
        self.condition_sets['enl'].append((day, condition, use_index))

    def exit_long_if(self, day, condition, use_index=False):
        """Adding an exit condition for long positions.

        :param int day: Day (relative to day 0 of signal completion) on which to check for the condition to be fulfilled.
        :param function condition: A function of x where x is a `pd.DataFrame` that returns a boolean `pd.Series` with the same index.
        :param bool, default=False use_index: If True, is being applied to the symbol df, else to the reference index df.

        For information on how conditions can be structured please refer to the user manual.

        :returns: None
        :rtype: NoneType
        """
        self.condition_sets['exl'].append((day, condition, use_index))

    def enter_short_if(self, day, condition, use_index=False):
        """Adding an entry condition for short positions.

        :param int day: Day (relative to day 0 of signal completion) on which to check for the condition to be fulfilled.
        :param function condition: A function of x where x is a `pd.DataFrame` that returns a boolean `pd.Series` with the same index.
        :param bool, default=False use_index: If True, is being applied to the symbol df, else to the reference index df.

        For information on how conditions can be structured please refer to the user manual.

        :returns: None
        :rtype: NoneType
        """
        self.condition_sets['ens'].append((day, condition, use_index))

    def exit_short_if(self, day, condition, use_index=False):
        """Adding an exit condition for short positions.

        :param int day: Day (relative to day 0 of signal completion) on which to check for the condition to be fulfilled.
        :param function condition: A function of x where x is a `pd.DataFrame` that returns a boolean `pd.Series` with the same index.
        :param bool, default=False use_index: If True, is being applied to the symbol df, else to the reference index df.

        For information on how conditions can be structured please refer to the user manual.

        :returns: None
        :rtype: NoneType
        """
        self.condition_sets['exs'].append((day, condition, use_index))

    def generate_signals(self):
        """Searches in data for signals based on all entry and exit conditions provided to this strategy.

        Discovered signals are annotated as columns 'entry_signals' and 'exit_signals' to each symbol df.
        In the 'entry_signals' column: 0 = no signal (do nothing), 1 = enter long, -1 = enter short.
        In the 'exit_signals' column: 0 = no signal (do nothing), -1 = exit long, 1 = exit short.

        :returns: The price data dict, with signals appended in the described columns.
        :rtype: dict

        For additional information please refer to the user manual.
        """

        if self.data is None or self.index is None:
            print('No data has been passed to this Strategy instance as self.data and/or self.index is None')
            return None

        for symbol, df in self.data.items():  # scanning for entry and exit conditions in data

            self.data[symbol]['score'] = 0
            self.data[symbol]['enl_signals'] = 0
            self.data[symbol]['exl_signals'] = 0
            self.data[symbol]['ens_signals'] = 0
            self.data[symbol]['exs_signals'] = 0

            # intermediate 'index_df' contains the same records as df
            index_df = self.index.loc[df.index.intersection(self.index.index)]

            # processing all signals...
            for k, v in self.condition_sets.items():

                boolean_array = pd.DataFrame()

                for i, condition in enumerate(v):
                    days, func, use_index = condition
                    if use_index:
                        boolean_array[i] = func(index_df).shift(-days)
                    else:
                        boolean_array[i] = func(df).shift(-days)

                signals = boolean_array.loc[boolean_array.T.all()].index.values
                self.data[symbol].loc[signals, f'{k}_signals'] = 1

        return self.data


class BackTest:
    """:class:`ndxtest.backtest.BackTest` runs backtests of trading strategies on an index level. Please refer
    to the user manual for examples how to use this class. The technical documentation focuses on listing
    instance variables and methods.

    :class:`ndxtest.backtest.BackTest` has more instance variables than listed below. The omitted variables are
    of no significance to users of ndxtest.

    :ivar str data_path: Absolute path to the data folder. Initial value: user input (__init__)
    :ivar list data_path_symbols: List of all symbols found in data\\lib. Initial value: []
    :ivar dict input_data: Dict containing the price data with ticker symbols as keys. Initial value: {}
    :ivar dict data: Copy of `input_data` with trading signals added to price data. Initial value: None
    :ivar pd.DataFrame input_index: Contains the price data of the reference index. Initial value: None
    :ivar pd.DataFrame index: Copy of `index` with computed indicators for signal generation. Initial value: None
    :ivar pd.DateRange dr: The range of dates between start and end date. Initial value: None
    :ivar pd.DateRange edr: Extended `dr` with additional data needed for calculating lagging indicators. Initial value: None
    :ivar bool dr_messages: If True, prints information on missing price data for `dr` and/or `edr` to the console. Initial value: None
    :ivar list trading_days: A list of all trading days between start and end date. Initial value: None
    :ivar list existing_symbols: All symbols included in the index during `dr` that exist in data\\lib. Initial value: None
    :ivar list missing_symbols: All symbols included in the index during `dr` that *do not* exist in data\\lib. Initial value: None
    :ivar pd.DataFrame eqc_df: A pd.DataFrame containing the equity curve of the backtest. Initial value: pd.DataFrame
    :ivar dict results: Contains high-level results (e.g. the winrate) of the backtest. Used to generate reports. Initial value: {}
    :ivar dict drawdown: Contains data used to calculate the (yearly) maximum drawdown. Initial value: {}
    :ivar dict exposure: Contains data on the net exposure to the market over time. Initial value: {}
    :ivar pd.DataFrame log_df: A pd.DataFrame containing a log of trades. Initial value: pd.DataFrame
    """

    def __init__(self, data_path):
        """Constructor Method. Connects the instance to the data folder. Fails if data folder not present.

        :param str data_path: The `data` directory or its parent directory.
        """

        self.data_path = connect(data_path, gspc_strict=True, hist_strict=True)
        self.data_path_symbols = [symbol[:-4] for symbol in os.listdir(self.data_path + 'lib\\') if '^' not in symbol]
        print(f"Setting the data_path to {data_path} was successful!")

        suffix = str(dt.datetime.now()).split('.')[0]
        suffix = suffix.replace(' ', '_')
        suffix = suffix.replace(':', '-')
        os.mkdir(self.data_path + 'backtest_' + suffix + '\\')
        self.output_path = self.data_path + 'backtest_' + suffix + '\\'

        self.input_data = {}
        self.data = None
        self.input_index = None
        self.index = None

        self.t0 = None
        self.runtime = None

        self.sd = None
        self.ed = None
        self.duration = None
        self.dr = None
        self.edr = None
        self.dr_messages = None
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
        self.alerts = None
        self.eqc = {}
        self.eqc_df = None
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

    @timeit_decorator
    def import_data(self, start_date, end_date, lag=200, date_range_messages=False):
        """Imports the necessary price data for the defined time period of the backtest.

        :param datetime.datetime or str start_date:
            The start date of the backtest. If weekend or US market closed, the next trading day will be set as start date.
        :param datetime.datetime or str end_date:
            The end date of the backtest. If weekend or US market closed, the next trading day will be set as end date.
        :param datetime.timedelta or int, default=200 lag:
            A timedelta that is added in front of the start_date. This is necessary for calculating indicators that
            depend on preceding price data such as moving averages among others.
        :param bool date_range_messages:
            If True, prints some extra information regarding missing price data for the backtest period to the console.

        :returns: None
        :rtype: NoneType
        """

        if not isinstance(start_date, dt.datetime):
            if not isinstance(start_date, str):
                raise TypeError("Parameter start_date must be of type datetime.datetime or str formatted 'YYYY-MM-DD'.")
            else:
                self.sd = dt.datetime.strptime(start_date, '%Y-%m-%d')
        else:
            self.sd = start_date

        if not isinstance(end_date, dt.datetime):
            if not isinstance(end_date, str):
                raise TypeError("Parameter start_date must be of type datetime.datetime or str formatted 'YYYY-MM-DD'.")
            else:
                self.ed = dt.datetime.strptime(end_date, '%Y-%m-%d')
        else:
            self.sd = end_date

        if not isinstance(lag, dt.timedelta):
            if not isinstance(lag, int):
                raise TypeError('Parameter lag must be of type datetime.timedelta or int.')
            else:
                lag = dt.timedelta(days=lag)

        self.duration = self.ed - self.sd
        self.dr = pd.date_range(self.sd, self.ed)
        self.edr = pd.date_range(self.sd - lag, self.ed)
        self.dr_messages = date_range_messages

        dtypes = {'symbol': str,
                  'open': np.float32,
                  'high': np.float32,
                  'low': np.float32,
                  'close': np.float32,
                  # 'volume': np.int64,
                  'dividends': np.float32,
                  'stock_splits': np.float32}
        self.input_index = pd.read_csv(self.data_path + 'lib\\^GSPC.csv', engine='c', dtype=dtypes,
                                       usecols=['date', 'symbol', 'open', 'high', 'low', 'close'],
                                       index_col='date', parse_dates=[0], dayfirst=True)
        self.input_index = self.input_index.loc[self.input_index.index.intersection(self.edr)]

        # self.alerts = "No alerts yet. Use the generate_signals or the scan_for_patterns method first."

        self.constituents = OrderedDict(sorted(constituents(self.sd, self.ed, lag, self.data_path).items()))

        self.existing_symbols = [s for s in self.constituents.keys() if s in self.data_path_symbols
                                 or ('*' in s and s[:-1] in self.data_path_symbols)]

        self.missing_symbols = [s for s in self.constituents.keys() if s not in self.existing_symbols]

        print(f'Constituents in time period: {len(self.constituents.keys())}. '
              f'Thereof not found in data path: {len(self.missing_symbols)}, {self.missing_symbols}')

        for symbol in self.existing_symbols:
            file = f"{self.data_path}lib\\{symbol[:-1] if '*' in symbol else symbol}.csv"
            dr, edr = self.constituents[symbol]['dr'], self.constituents[symbol]['edr']

            with open(file) as f:
                next(f)
                first = next(f).split(',')[0]
                first = dt.datetime.strptime(first, '%Y-%m-%d')

                skip = ((self.sd - first).days * 0.65).__floor__() - lag.days

            df = pd.read_csv(file, engine='c', dtype=dtypes, skiprows=range(1, skip), index_col='date',
                             parse_dates=['date'])
            df = df.loc[df.index.intersection(edr)]
            df['symbol'] = symbol

            if self.dr_messages:
                missing_records = df.index.symmetric_difference(edr.intersection(self.input_index.index))
                if any(missing_records):
                    print(f'{symbol}: {len(missing_records)} missing records in extended date range.')

            if not df.empty:
                self.input_data[symbol] = df

    @timeit_decorator
    def generate_signals(self, strategy):
        """Accepts an instance of the :class:`ndxtest.backtest.Strategy` class. Generates and oders the trading signals.

        :param ndxtest.backtest.Strategy strategy:
            A trading strategy built with the :class:`ndxtest.backtest.Strategy` class.

        :returns: None
        :rtype: NoneType
        """

        if not isinstance(strategy, Strategy):
            raise TypeError("Argument `strategy` must be an instance of ndxtest.backtest.Strategy.")

        strategy.data = self.input_data.copy()
        strategy.index = self.input_index.copy()
        self.data = strategy.generate_signals()

        for symbol, df in self.data.items():
            self.data[symbol] = df.loc[df.index.intersection(self.constituents[symbol]['dr'])]

        self.index = self.input_index.loc[self.input_index.index.intersection(self.dr)]
        self.index['d%change'] = self.index.close.pct_change() * 100
        self.index['c%change'] = ((self.index.close / self.index.close[0]) - 1) * 100
        self.index = self.index.round(2)

        self.trading_days = self.index.index.tolist()

        concat_data = pd.concat(self.data.values())
        concat_data.drop(columns=['high', 'low', 'close', 'volume', 'dividends', 'stock_splits'], inplace=True)
        dict_data = {}

        # self.alerts = concat_data.loc[(concat_data.index == self.trading_days[-1]) &
        #                               ((concat_data.entry_signals != 0) | (concat_data.exit_signals != 0))]

        for symbol, df in concat_data.groupby('symbol'):
            # trades will be executed on the next day
            df['enl_signals'] = df['enl_signals'].shift(1)
            df['exl_signals'] = df['exl_signals'].shift(1)
            df['ens_signals'] = df['ens_signals'].shift(1)
            df['exs_signals'] = df['exs_signals'].shift(1)

            # adding signals to exit on the last day
            df.loc[df.index[-1], ['score', 'enl_signals', 'exl_signals', 'ens_signals', 'exs_signals']] = \
                [0, 0, -2, 0, -2]  # check!

            # dropping NaN values
            df.dropna(inplace=True)
            # dropping emtpy rows
            dict_data[symbol] = df.loc[(df.enl_signals != 0) | (df.exl_signals != 0) | (df.ens_signals != 0) | (df.exs_signals != 0)]

        for date, values in pd.concat(dict_data.values()).groupby(level=0):
            # date, df: symbol, score, enl_signals, exl_signals, ens_signals, exs_signals

            self.signals[date] = values.to_dict(orient='records')
            self.signals[date] = sorted(list(self.signals[date]), key=lambda signal: signal['score'], reverse=True)

    @timeit_decorator
    def eval_signals(self, strategy):
        """This function searches the imported data for the signals specified by the strategy. It creates a .xlsx report
        about the immediate price action following the signals. It helps evaluating the viability of the signals
        generated by the strategy.

        :param ndxtest.backtest.Strategy strategy: A ndxtest.backtest.Strategy object.
        """

        if not isinstance(strategy, Strategy):
            raise TypeError("Argument `strategy` must be an instance of ndxtest.backtest.Strategy.")

        strategy_signals = {'entry_long_signals': strategy.condition_sets['enl'],
                            'exit_long_signals': strategy.condition_sets['exl'],
                            'entry_short_signals': strategy.condition_sets['ens'],
                            'exit_short_signals': strategy.condition_sets['exs']}

        with pd.ExcelWriter(self.output_path + 'signal_stats.xlsx') as writer:

            for signal_type, list_of_conditions in strategy_signals.items():
                results = {}

                for symbol, df in self.input_data.items():
                    signals = pd.DataFrame()

                    for i, condition in enumerate(list_of_conditions):
                        delta_days, func, index = condition
                        if not index:
                            signals[i] = func(df).shift(-delta_days)
                        else:
                            signals[i] = func(self.input_index).shift(-delta_days)

                    cds = signals.loc[signals.T.all()].index.values  # cds = completion days of pattern (=0)

                    results[symbol] = pd.DataFrame(data={'cd': cds,
                                                         'symbol': symbol,
                                                         'ep': df.shift(-1).loc[cds, 'open'],  # entry price
                                                         '+1c': df.shift(-1).loc[cds, 'close'],
                                                         '+1c_gt_ep': df.shift(-1).loc[cds, 'open'] < df.shift(-1).loc[
                                                             cds, 'close'],
                                                         '+2o': df.shift(-2).loc[cds, 'open'],
                                                         '+2o_gt_ep': df.shift(-1).loc[cds, 'open'] < df.shift(-2).loc[
                                                             cds, 'open'],
                                                         '+2c': df.shift(-2).loc[cds, 'close'],
                                                         '+2c_gt_ep': df.shift(-1).loc[cds, 'open'] < df.shift(-2).loc[
                                                             cds, 'close'],
                                                         '+3o': df.shift(-3).loc[cds, 'open'],
                                                         '+3o_gt_ep': df.shift(-1).loc[cds, 'open'] < df.shift(-3).loc[
                                                             cds, 'open'],
                                                         '+3c': df.shift(-3).loc[cds, 'close'],
                                                         '+3c_gt_ep': df.shift(-1).loc[cds, 'open'] < df.shift(-3).loc[
                                                             cds, 'close']
                                                         })

                d = pd.concat(results, ignore_index=True).sort_values(by='cd')

                if len(d.index) == 0:
                    print(f'No {signal_type} found!')
                    continue
                else:
                    d.dropna(inplace=True)

                    cols = [('+1c', '+1c_gt_ep'), ('+2o', '+2o_gt_ep'), ('+2c', '+2c_gt_ep'), ('+3o', '+3o_gt_ep'),
                            ('+3c', '+3c_gt_ep')]
                    stats = {'metric': ['Total count:',
                                        'Up count:',
                                        'Down count:',
                                        'Up %:',
                                        'Down %:',
                                        'Avg. % change:',
                                        'Med. % change:',
                                        'Std. %:',
                                        'Up avg. % change:',
                                        'Up med. % change:',
                                        'Up std. %:',
                                        'Up % change min:',
                                        'Up % change max:',
                                        'Down avg. % change:',
                                        'Down med. % change:',
                                        'Down std. %:',
                                        'Down % change min:',
                                        'Down % change max:',
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

                    d.to_excel(writer, sheet_name=signal_type, index=False)
                    pd.DataFrame(data=stats).to_excel(writer, sheet_name=signal_type + '_stats', index=False)

    @timeit_decorator
    def run_backtest(self, commission=.001, max_positions=10, initial_equity=10000.00,
                     max_trade_duration=None, stoploss=None, detailed_eqc=True):
        """Executes the backtest and creates several logs in the meantime.

        :param float, default=.001 commission:
            Commission that is paid upon entering/exiting positions.
        :param int, default=10 max_positions:
            Maximum number of positions (slots) in the portfolio.
        :param float, deault=10000.00 initial_equity:
            The initial capital to start with.
        :param int, default=None max_trade_duration:
            Maximum number of days before positions will be closed. Will close positions independent of any signals provided by the strategy.
        :param float, deaulft=None stoploss:
            The maximum % (e.g. 0.05) of adverse price movement before positions will be closed. Will close positions indepent of signals provided by the strategy.
        :param bool, default=True detailed_eqc:
            If True, calculates the ohlc market value of the portfolio daily during the backtest period. Significantly increases time for computation of backtest. Set this to false for quick testing.

        :returns: None
        :rtype: None
        """

        if self.output_path is None:
            suffix = str(dt.datetime.now()).split('.')[0]
            suffix = suffix.replace(' ', '_')
            suffix = suffix.replace(':', '-')
            self.output_path = self.data_path + suffix + '\\'

        self.commission = commission
        self.max_positions = max_positions
        self.initial_equity = initial_equity
        self.max_trade_duration = max_trade_duration
        self.stoploss = stoploss
        self.exposure = {}

        p = Portfolio(max_positions=max_positions, initial_equity=initial_equity, commission=commission)

        for date in self.trading_days:
            max_trade_duration_signals = []
            stoploss_signals = []

            if max_trade_duration is not None and p.positions:
                current_positions = list(p.long_positions.values()) + list(p.short_positions.values())

                max_trade_duration_violated = \
                    list(filter(lambda position: position['entry_date'] <= date - dt.timedelta(days=max_trade_duration),
                                current_positions))

                max_trade_duration_signals = [{'symbol': position['symbol'],
                                               'enl_signals': 0,
                                               'exl_signals': -2,
                                               'ens_signals': 0,
                                               'exs_signals': -2,
                                               'score': 0,
                                               'open': self.data[position['symbol']].loc[date, 'open']}
                                              for position in max_trade_duration_violated]

            if stoploss is not None and p.positions:
                long_positions, short_positions = list(p.long_positions.values()), list(p.short_positions.values())

                long_stoploss_violated = list(filter(lambda position:
                                                     position['entry_price'] * (1 - stoploss) >
                                                     self.data[position['symbol']].loc[date, 'open'], long_positions))

                short_stoploss_violated = list(filter(lambda position:
                                                      position['entry_price'] * (1 + stoploss) <
                                                      self.data[position['symbol']].loc[date, 'open'], short_positions))

                stoploss_signals = [{'symbol': position['symbol'],
                                     'enl_signals': 0,
                                     'exl_signals': -2,
                                     'ens_signals': 0,
                                     'exs_signals': -2,
                                     'score': 0,
                                     'open': self.data[position['symbol']].loc[date, 'open']}
                                    for position in long_stoploss_violated + short_stoploss_violated]

            self.signals[date] += max_trade_duration_signals
            self.signals[date] += stoploss_signals

            # processing exit signals
            if self.signals[date]:

                for s in [s for s in self.signals[date] if s['symbol'] in p.positions()]:

                    # exiting short position
                    if s['exs_signals'] in {1, -2} and s['symbol'] in p.short_positions:
                        p.long(data={'symbol': s['symbol'],
                                     'exit_score': s['score'],
                                     'exit_price': s['open'],
                                     'exit_date': date})

                    # exiting long position
                    if s['exl_signals'] in {1, -2} and s['symbol'] in p.long_positions:
                        p.short(data={'symbol': s['symbol'],
                                      'exit_score': s['score'],
                                      'exit_price': s['open'],
                                      'exit_date': date})

                if date == self.trading_days[-1]:
                    self.signals[date] = []
                else:
                    self.signals[date] = list(filter(lambda signal: signal['symbol'] not in p.positions(), self.signals[date]))

                # processing entry signals
                for s in self.signals[date]:

                    # entering long
                    if s['enl_signals'] == 1 and s['symbol'] and p.free_slot():
                        s['nshares'] = p.calculate_nshares(s['open'])
                        if s['nshares'] > 0:
                            p.long(data={'symbol': s['symbol'],
                                         'signal': 1,
                                         'entry_score': s['score'],
                                         'entry_price': s['open'],
                                         'nshares': s['nshares'],
                                         'entry_date': date})
                            continue  # if enl and ens signals are present enl has precedence

                    # entering short
                    if s['ens_signals'] == 1 and s['symbol'] and p.free_slot():
                        s['nshares'] = -1 * p.calculate_nshares(s['open'])
                        if s['nshares'] < 0:
                            p.short(data={'symbol': s['symbol'],
                                          'signal': 1,
                                          'entry_score': s['score'],
                                          'entry_price': s['open'],
                                          'nshares': s['nshares'],
                                          'entry_date': date})

            # logging the "market exposure" statistic
            self.exposure[date] = p.current_exposure()

            if detailed_eqc:
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

        if detailed_eqc:
            self.eqc_df = pd.DataFrame.from_dict(data=self.eqc, orient='index')
            self.eqc_df['d%change'] = self.eqc_df.close.pct_change() * 100
            self.eqc_df['c%change'] = (((self.eqc_df.close / initial_equity) - 1) * 100)
            self.eqc_df = self.eqc_df.round(2)
            self.eqc_df.to_csv(self.output_path + 'equity_curve.csv')
        else:
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

    @timeit_decorator
    def report(self):
        """Generates .xlsx and .pdf reports of the last run backtest.

        :returns: None
        :rtype: NoneType
        """

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

        # pdf.cell(w=80, align='', txt=f'Parameters:', ln=0)
        # pdf.cell(w=80, align='', txt=f'{self.best_parameters if self.opt else "not optimized"}', ln=1)
        pdf.cell(w=80, align='', txt=f'Start date:', ln=0)
        pdf.cell(w=80, align='', txt=f'{self.sd.date()}', ln=1)
        pdf.cell(w=80, align='', txt=f'End date:', ln=0)  #
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
        pdf.cell(w=80, align='', txt=f"{str(self.results['bm_perf'])[:5]}", ln=1)
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

        pdf.output(self.output_path + 'backtest_report.pdf', 'F')
        os.remove(self.output_path + 'f1.png')

    def query_missing_records(self, date):
        """Queries, whether any symbols lack price data for a specific date. (for debugging the library)

        :param datetime.datetime or str date: If string it must be 'YYYY-MM-DD' formatted.

        :returns: None
        :rtype: NoneType
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
