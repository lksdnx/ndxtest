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

    k = {'A': a, 'AMZN': amzn}

    bt = BackTest(data_path='data\\lib\\', start_date=dt.datetime(2020, 1, 1), end_date=dt.datetime(2022, 6, 1),
                  lag=dt.timedelta(days=0), runtime_messages=True, date_range_messages=False)

    bt.setup_search(pattern=[(-2, gap_down_wick, False), (-1, gap_down_wick, False), (0, gap_down_wick, False)])

    # print(bullish_pin_bar(bt.input_data['ILMN']))

    # bt.generate_signals(strategy=str1)
    # bt.run(long_only=False, max_positions=10, commission=.001, max_trade_duration=None, stoploss=0.20, eqc_method='full')
    # bt.report()
    # bt.plot_ticker('BBWI')

    # se = "df = pd.DataFrame(data=np.random.randn(10000, 4), columns=['open', 'high', 'low', 'close'])"
    # s = "bbands(df)"
    # n = 1000
    # print(timeit.timeit(stmt=s, setup=se, number=n, globals=globals())/n)
