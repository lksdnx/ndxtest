import os
from distutils.dir_util import copy_tree
import pandas as pd
import numpy as np
import datetime as dt
import openpyxl
from spytools.backtest import BackTest
from spytools.utils import *
from spytools.strategies import str1
from indicators.candlesticks import bullish_pin_bar, bearish_pin_bar, bullish_pib_pattern, bearish_pib_pattern
from indicators.price_action import *

path = "data\\lib\\"

if __name__ == "__main__":
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)

    a = pd.read_csv(path + 'A.csv', index_col='date', parse_dates=True)
    amzn = pd.read_csv(path + 'AMZN.csv', index_col='date', parse_dates=True)

    d = {'A': a, 'AMZN': amzn}


    def setup_search(pattern, data):
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
        r = {}  # results

        if isinstance(pattern, list):

            for s, df in data.items():  # s = symbol

                signals = pd.DataFrame()

                for i, element in enumerate(pattern):
                    delta_days, func, index = element  # how will funcs with arguments be handled?
                    signals[delta_days] = func(df).shift(-delta_days)

                cds = signals.loc[signals.T.all()].index.values  # cds = completion days of pattern (=0)

                r[s] = pd.DataFrame(data={'cd': cds,
                                          'symbol': s,
                                          'ep': df.shift(-1).loc[cds, 'open'],  # entry price
                                          'c': df.shift(-1).loc[cds, 'close'],
                                          'c_gt_ep': df.shift(-1).loc[cds, 'open'] < df.shift(-1).loc[cds, 'close'],
                                          '+1o': df.shift(-2).loc[cds, 'open'],
                                          '+1o_gt_ep': df.shift(-1).loc[cds, 'open'] < df.shift(-2).loc[cds, 'open'],
                                          '+1c': df.shift(-2).loc[cds, 'close'],
                                          '+1c_gt_ep': df.shift(-1).loc[cds, 'open'] < df.shift(-2).loc[cds, 'close'],
                                          '+2o': df.shift(-3).loc[cds, 'open'],
                                          '+2o_gt_ep': df.shift(-1).loc[cds, 'open'] < df.shift(-3).loc[cds, 'open'],
                                          '+2c': df.shift(-3).loc[cds, 'close'],
                                          '+2c_gt_ep': df.shift(-1).loc[cds, 'open'] < df.shift(-3).loc[cds, 'close']
                                          }
                                    )

            data = pd.concat(r, ignore_index=True).sort_values(by='cd')
            data.dropna(inplace=True)

            cols = [('c', 'c_gt_ep'), ('+1o', '+1o_gt_ep'), ('+1c', '+1c_gt_ep'), ('+2o', '+2o_gt_ep'), ('+2c', '+2c_gt_ep')]
            stats = {}
            for price_col, count_col in cols:
                f, t = data[count_col].value_counts().sort_index().tolist()
                stats[count_col] = [f'True: {t}', f'False: {f}', f'Rate: {round((t / (f+t) * 100), 2)}']

            with pd.ExcelWriter('pattern_stats.xlsx') as writer:
                data.to_excel(writer, sheet_name='data', index=False)
                pd.DataFrame(data=stats).to_excel(writer, sheet_name='stats')


    bt = BackTest(data_path='data\\lib\\', start_date=dt.datetime(2015, 1, 1), end_date=dt.datetime(2022, 6, 1),
                  lag=dt.timedelta(days=0), runtime_messages=True, date_range_messages=False)

    setup_search(pattern=[(-2, gap_down_wick, False), (-1, gap_down_wick, False), (0, gap_down_wick, False)],
                 data=bt.input_data)




    # print(bullish_pin_bar(bt.input_data['ILMN']))

    # bt.generate_signals(strategy=str1)
    # bt.run(long_only=False, max_positions=10, commission=.001, max_trade_duration=None, stoploss=0.20, eqc_method='full')
    # bt.report()
    # bt.plot_ticker('BBWI')

    # se = "df = pd.DataFrame(data=np.random.randn(10000, 4), columns=['open', 'high', 'low', 'close'])"
    # s = "bbands(df)"
    # n = 1000
    # print(timeit.timeit(stmt=s, setup=se, number=n, globals=globals())/n)
