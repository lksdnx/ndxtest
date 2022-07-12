from pathlib import Path
import os
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import pandas as pd
import decimal
import itertools
from collections import OrderedDict, defaultdict
from spytools.utils import constituents
import time
import datetime as dt
import mplfinance as mpf
import random
import seaborn as sns
from fpdf import FPDF


class Portfolio:
    def __init__(self, max_positions, initial_equity, commission):
        self.max_positions = max_positions
        self.cash = initial_equity
        self.commission = commission
        self.invested_capital, self.commission_paid, self.number_of_trades = 0, 0, 0
        self.cash_from_short_positions = {}
        self.long_positions, self.short_positions = {}, {}
        self.logdict = OrderedDict()

    def positions(self):
        return set(list(self.long_positions.keys()) + list(self.short_positions.keys()))

    def number_of_positions(self):
        return len(self.positions())

    def free_slot(self):
        return True if self.number_of_positions() < self.max_positions else False

    def current_exposure(self):
        return {'n_long': len(self.long_positions.keys()),
                'n_short': len(self.short_positions.keys()),
                'n_free': self.max_positions - self.number_of_positions(),
                'net_exposure': len(self.long_positions.keys()) - len(self.short_positions.keys())}

    def cash_from_shorts(self):
        return sum(self.cash_from_short_positions.values())

    def calculate_nshares(self, entry_price):
        return (((self.cash - self.cash_from_shorts()) /
                 (self.max_positions - self.number_of_positions())) / entry_price).__floor__()

    def describe_current_status(self, date):
        print(f'Stats of {self} as of {date}:')
        print(f'Number of Positions: {self.number_of_positions()}. '
              f'Long: {list(self.long_positions.keys())}, '
              f'Short: {list(self.short_positions.keys())}')
        print(f'Cash: {self.cash.__floor__()}. Cash from shorts: {self.cash_from_shorts().__floor__()}. '
              f'Invested Capital: {self.invested_capital.__floor__()}.')
        print(f'Total Market Value: {(self.invested_capital + self.cash + self.cash_from_shorts()).__floor__()}')
        print('\n')

    def create_log_entry(self, entry_data=None, exit_data=None):
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
             'p.sh_cash': self.cash_from_shorts(),
             'p.invested_capital': self.invested_capital,
             'market_value': (self.invested_capital + self.cash + self.cash_from_shorts())}
        return d

    def create_log_df(self):
        return pd.DataFrame.from_dict(data=self.logdict, orient='index').round(2)

    def long(self, data):  # closing short position has priority
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
    def __init__(self, data_path, start_date=dt.datetime(2015, 9, 1), end_date=dt.datetime(2021, 9, 1),
                 lag=dt.timedelta(days=200), runtime_messages=True, date_range_messages=False):
        """The __init__ method of the BackTest class is responsible for importing price data. The data is stored in the
        self.input_data field as a dictionary with following structure: {'symbol': data as a pd.Dataframe}."""
        self.runtime_messages = runtime_messages
        self.t0, self.tr, self.runtime = time.time(), None, None
        self.sd, self.ed, self.duration = start_date, end_date, end_date - start_date
        self.dr, self.edr = pd.date_range(start_date, end_date), pd.date_range(start_date - lag, end_date)
        self.trading_days = None

        float32_cols = {'open': np.float32, 'high': np.float32, 'low': np.float32, 'close': np.float32}

        self.input_data = {}
        self.data = None

        self.input_benchmark = pd.read_csv(f'{data_path}^GSPC.csv', engine='c', dtype=float32_cols,
                                           index_col='date', parse_dates=[0], dayfirst=True)
        self.input_benchmark.drop(columns=['dividends', 'stock_splits'], inplace=True)
        self.input_benchmark = self.input_benchmark.loc[self.input_benchmark.index.intersection(self.edr)]
        self.benchmark = None

        self.signals = defaultdict(list)
        self.alerts = "No alerts yet. Use the generate_signals or the scan_for_patterns method first."

        self.commission = None
        self.max_positions = None
        self.initial_equity = None
        self.max_trade_duration = None
        self.stoploss = None
        self.entry_mode = None
        self.log_df, self.equity_curve_df = pd.DataFrame(), pd.DataFrame()

        self.eqc = {}
        self.results = {}
        self.drawdown = {}
        self.exposure = {}
        self.correlations = {}

        self.opt, self.optimization_results, self.best_parameters = False, None, None
        self.parameters, self.parameter_permutations = {}, []

        self.data_path_symbols = [str(file)[len(data_path):-4]
                                  for file in Path(data_path).glob('*.csv') if '^' not in str(file)]

        self.constituents = OrderedDict(
            sorted(constituents(data_path=data_path, start_date=start_date, end_date=end_date, lag=lag).items()))

        self.existing_symbols = [s for s in self.constituents.keys() if s in self.data_path_symbols
                                 or ('*' in s and s[:-1] in self.data_path_symbols)]

        self.missing_symbols = [s for s in self.constituents.keys() if s not in self.existing_symbols]

        print(f'Importing data...')
        print(f'Constituents in time period: {len(self.constituents.keys())}. '
              f'Thereof not found in data path: {len(self.missing_symbols)}. ')
        print(f'Missing symbols: {self.missing_symbols}')

        for symbol in self.existing_symbols:
            dr, edr = self.constituents[symbol]['dr'], self.constituents[symbol]['edr']

            with open(f"{data_path}{symbol[:-1] if '*' in symbol else symbol}.csv") as f:
                next(f)
                first = next(f).split(',')[0]
                first = dt.datetime.strptime(first, '%Y-%m-%d')
                skip = ((start_date - first).days * 0.65).__floor__() - lag.days

            df = pd.read_csv(f"{data_path}{symbol[:-1] if '*' in symbol else symbol}.csv", engine='c',
                             dtype=float32_cols, skiprows=range(1, skip), index_col='date', parse_dates=['date'])
            df = df.loc[df.index.intersection(edr)]
            df['symbol'] = symbol

            if date_range_messages:
                missing_records = df.index.symmetric_difference(edr.intersection(self.input_benchmark.index))
                if any(missing_records):
                    print(f'{symbol}: {len(missing_records)} missing records in extended date range.')

            if not df.empty:
                self.input_data[symbol] = df

        if self.runtime_messages:
            print(f'Data imported:                              ...{(time.time() - self.t0).__round__(2)} sec elapsed.')

    def query_missing_records(self, datestring):
        date = dt.datetime.strptime(datestring, '%Y-%m-%d')
        if date.isoweekday() in {6, 7}:
            print(f'{datestring} is not a weekday. Call function with weekdays only.')
        elif date < self.sd:
            print(f'{datestring} is not within the backtest period. The record might be available but not imported.')
        else:
            for k, v in self.input_data.items():
                if date not in v.index:
                    if date in self.constituents[k]['dr']:
                        print(f'Data for {k} is missing the record for {date}.')

    def setup_search(self, pattern):
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

            for symbol, df in self.input_data.items():

                signals = pd.DataFrame()

                for i, element in enumerate(pattern):
                    delta_days, func, index = element  # how will funcs with arguments be handled?
                    signals[i] = func(df).shift(-delta_days)

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
                                                     }
                                               )

            d = pd.concat(results, ignore_index=True).sort_values(by='cd')
            if len(d.index) == 0:
                print(d)
                print('No instances of pattern found!')
                exit()
            d.dropna(inplace=True)
            rnd = pd.DataFrame.from_dict(random_results, orient='index').sort_values(by='cd')
            rnd.dropna(inplace=True)

            cols = [('c', 'c_gt_ep'), ('+1o', '+1o_gt_ep'), ('+1c', '+1c_gt_ep'), ('+2o', '+2o_gt_ep'),
                    ('+2c', '+2c_gt_ep')]
            stats = {'metric': ['T count:',
                                'F count:',
                                'T %:',
                                'Avg. % change:',
                                'Med. % change:',
                                'T avg. % change:',
                                'T med. % change:',
                                'T % change min:',
                                'T % change max:',
                                'F avg. % change:',
                                'F med. % change:',
                                'F % change min:',
                                'F % change max:',
                                ]}
            rnd_stats = {'metric': ['T count:',
                                    'F count:',
                                    'T %:',
                                    'Avg. % change:',
                                    'Med. % change:',
                                    'T avg. % change:',
                                    'T med. % change:',
                                    'T % change min:',
                                    'T % change max:',
                                    'F avg. % change:',
                                    'F med. % change:',
                                    'F % change min:',
                                    'F % change max:',
                                    ]}

            # making stats from the experimental data
            for price_col, count_col in cols:
                fc, tc = d[count_col].value_counts().sort_index().tolist()
                pct_change = ((d[price_col] / d['ep']) - 1) * 100
                t_pct_change = ((d.loc[d[count_col], price_col] / d.loc[d[count_col], 'ep']) - 1) * 100
                f_pct_change = ((d.loc[~d[count_col], price_col] / d.loc[~d[count_col], 'ep']) - 1) * 100
                stats[count_col] = [tc,
                                    fc,
                                    round((tc / (fc + tc) * 100), 2),
                                    round(pct_change.mean(), 2),
                                    round(pct_change.median(), 2),
                                    round(t_pct_change.mean(), 2),
                                    round(t_pct_change.median(), 2),
                                    round(t_pct_change.min(), 2),
                                    round(t_pct_change.max(), 2),
                                    round(f_pct_change.mean(), 2),
                                    round(f_pct_change.median(), 2),
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
                stats[count_col + '_rnd'] = [tc,
                                             fc,
                                             round((tc / (fc + tc) * 100), 2),
                                             round(pct_change.mean(), 2),
                                             round(pct_change.median(), 2),
                                             round(t_pct_change.mean(), 2),
                                             round(t_pct_change.median(), 2),
                                             round(t_pct_change.min(), 2),
                                             round(t_pct_change.max(), 2),
                                             round(f_pct_change.mean(), 2),
                                             round(f_pct_change.median(), 2),
                                             round(f_pct_change.min(), 2),
                                             round(f_pct_change.max(), 2)
                                             ]

            with pd.ExcelWriter('pattern_stats.xlsx') as writer:
                d.to_excel(writer, sheet_name='pattern', index=False)
                rnd.to_excel(writer, sheet_name='random', index=False)
                pd.DataFrame(data=stats).to_excel(writer, sheet_name='stats_pattern_vs_random', index=False)

    def generate_signals(self, strategy, parameters=None):
        """Write proper docstring!"""
        t1, self.tr = time.time(), time.time()
        self.data = strategy(self.input_data, self.input_benchmark, parameters=parameters)
        if self.runtime_messages:
            print(f'Signals generated:                          ...{(time.time() - t1).__round__(2)} sec elapsed.')

        t1 = time.time()
        for symbol, df in self.data.items():
            self.data[symbol] = df.loc[df.index.intersection(self.constituents[symbol]['dr'])]

        self.benchmark = self.input_benchmark.loc[self.input_benchmark.index.intersection(self.dr)]
        self.benchmark['d%change'] = self.benchmark.close.pct_change() * 100
        self.benchmark['c%change'] = ((self.benchmark.close / self.benchmark.close[0]) - 1) * 100
        self.benchmark = self.benchmark.round(2)

        self.trading_days = self.benchmark.index.tolist()

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

    def execute_signals(self, long_only=False, commission=.001, max_positions=10, initial_equity=10000.00,
                        max_trade_duration=None, stoploss=None, entry_mode='open', eqc_method='approx'):
        """Entry Signals: 1 = enter long position, -1 = enter short position,
        Exit Signals: 1 = exit short position, -1 = exit long position, -2 = exit long or short position"""
        self.commission = commission
        self.max_positions = max_positions
        self.initial_equity = initial_equity
        self.max_trade_duration = max_trade_duration
        self.stoploss = stoploss
        self.exposure = {}
        self.entry_mode = entry_mode

        p = Portfolio(max_positions=max_positions, initial_equity=initial_equity, commission=commission)

        t1 = time.time()
        if self.runtime_messages:
            print(f'Running backtest...')

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
                data = {'open': p.cash + p.cash_from_shorts() + op, 'high': p.cash + p.cash_from_shorts() + hi,
                        'low': p.cash + p.cash_from_shorts() + lo, 'close': p.cash + p.cash_from_shorts() + cl}
                self.eqc[date] = data

        self.exposure = pd.DataFrame.from_dict(data=self.exposure, orient='index', dtype=np.int8)

        self.log_df = p.create_log_df()
        self.log_df.to_csv('output\\tradelog.csv')
        self.log_df = self.log_df.loc[self.log_df['exit_date'] != 0]

        if eqc_method == 'full':
            self.equity_curve_df = pd.DataFrame.from_dict(data=self.eqc, orient='index')
            self.equity_curve_df['d%change'] = self.equity_curve_df.close.pct_change() * 100
            self.equity_curve_df['c%change'] = (((self.equity_curve_df.close / initial_equity) - 1) * 100)
            self.equity_curve_df = self.equity_curve_df.round(2)
            self.equity_curve_df.to_csv('output\\equity_curve.csv')

        if eqc_method == 'approx':
            x = self.log_df.drop_duplicates(subset=['exit_date'], keep='last')
            x = x.set_index('exit_date')['market_value']
            self.equity_curve_df = pd.DataFrame(data={'close': np.nan}, index=self.benchmark.index)
            self.equity_curve_df.loc[x.index, 'close'] = x
            self.equity_curve_df.fillna(method="ffill", inplace=True)
            self.equity_curve_df.fillna(self.initial_equity, inplace=True)
            self.equity_curve_df['d%change'] = self.equity_curve_df.close.pct_change() * 100
            self.equity_curve_df['c%change'] = (((self.equity_curve_df.close / initial_equity) - 1) * 100)
            self.equity_curve_df = self.equity_curve_df.round(2)
            self.equity_curve_df.to_csv('output\\equity_curve.csv')

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
                        'bm_perf': self.benchmark['c%change'][-1],
                        'bm_ann_perf': (((1 + (
                                (self.benchmark.close[-1] - self.benchmark.close[0]) / self.benchmark.close[0]))
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
                self.execute_signals(long_only=True,
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

            plt.savefig("output\\f3.png", dpi=None, facecolor='w', edgecolor='w')

            self.best_parameters = {f'p{i + 1}': val for i, val in enumerate(self.optimization_results.idxmax())}
            self.opt = True

            if run_best:
                self.generate_signals(strategy, self.best_parameters)
                self.execute_signals(long_only=True,
                                     commission=self.commission,
                                     max_positions=self.max_positions,
                                     initial_equity=self.initial_equity,
                                     max_trade_duration=self.max_trade_duration,
                                     stoploss=self.stoploss,
                                     eqc_method='full')

    def report(self):

        fig1, axs1 = plt.subplots(3, 1, figsize=(9, 6))
        axs1[0].plot(self.benchmark['c%change'].index, self.benchmark['c%change'], color='blue', label='Benchmark')
        axs1[0].plot(self.equity_curve_df['c%change'].index, self.equity_curve_df['c%change'], color='black',
                     label='Backtest')
        axs1[0].set_xticks([])
        axs1[0].set_xticklabels([])
        axs1[0].set_ylabel('Cumulative % Change')
        axs1[0].grid(axis='y')
        axs1[0].legend()
        axs1[1].plot(self.drawdown['daily'].index, self.drawdown['daily'], color='black', label='Drawdown')
        axs1[1].plot(self.drawdown['daily_max'].index, self.drawdown['daily_max'], color='red',
                     label='Max. 1Yr Drawndown')
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

        plt.savefig("output\\f1.png", dpi=None, facecolor='w', edgecolor='w')

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

        plt.savefig("output\\f2.png", dpi=None, facecolor='w', edgecolor='w')

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
        pdf.image('output\\f1.png', x=15, y=25, w=180, h=120)

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
            shape = img.imread('output\\f3.png').shape
            pdf.image('output\\f3.png', x=10, y=20, w=shape[0] * (160 / shape[0]), h=shape[1] * (160 / shape[1]))

        pdf.output('output\\test.pdf', 'F')

        os.remove('output\\f1.png')
        os.remove('output\\f2.png')
        if self.opt:
            os.remove('output\\f3.png')

    def plot_ticker(self, symbol=None):
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
