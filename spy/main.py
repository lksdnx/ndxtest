import os
import pandas as pd
import numpy as np
import datetime as dt
from backtest.backtest import BackTest
from indicators.candlesticks import bullish_pin_bar
from indicators.price_action import *
import setuptools

path = "data\\lib\\"

if __name__ == "__main__":
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)

    bt = BackTest(data_path='data\\lib\\', start_date=dt.datetime(2021, 1, 1), end_date=dt.datetime(2022, 6, 1),
                  lag=dt.timedelta(days=0), runtime_messages=True, date_range_messages=False)

    bt.setup_search(pattern=[(-2, gap_down_wick, False),
                             (-1, gap_down_wick, False),
                             (0, bullish_pin_bar, False),
                             (1, gap_up_body, True)])

    # se = "df = pd.DataFrame(data=np.random.randn(10000, 4), columns=['open', 'high', 'low', 'close'])"
    # s = "bbands(df)"
    # n = 1000
    # print(timeit.timeit(stmt=s, setup=se, number=n, globals=globals())/n)
