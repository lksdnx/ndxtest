"""TUTORIAL"""
import datetime as dt
import numpy as np
import pandas as pd
from ndxtest.utils import LibManager
from ndxtest.backtest import BackTest, Strategy
from ndxtest.indicators import *


if __name__ == "__main__":

    # lm = LibManager('C:\\Users\\user\\absolute\\location\\of\\data')
    # lm.lib_update(symbols='ANTM')

    # s1
    s1 = Strategy()
    s1.enter_long_if(0, lambda x: crossover(50, rsi(x), 'bullish'))
    s1.enter_long_if(0, lambda x: x.close > sma(x, 20))
    s1.exit_long_if(0, lambda x: crossover(sma(x, 20), x.close, 'bearish'))

    bt1 = BackTest('C:\\Users\\user\\absolute\\location\\of\\data')
    bt1.import_data(start_date='2019-06-01', end_date='2021-06-01')
    bt1.generate_signals(s1)
    bt1.run_backtest()
    bt1.report()

    # s2
    s2 = Strategy()
    s2.enter_long_if(0, lambda x: crossover(50, rsi(x), 'bullish'))
    s2.enter_long_if(0, lambda x: x.close > sma(x, 20))
    s2.enter_long_if(1, lambda x: x.open > prev(x.close))
    s2.exit_long_if(0, lambda x: crossover(sma(x, 70), sma(x, 20), 'bearish'))

    s2.enter_short_if(0, lambda x: crossover(40, rsi(x), 'bearish'))
    s2.enter_short_if(0, lambda x: roc(x, 5) < -0.08, True)

    s2.exit_short_if(0, lambda x: roc(x, 10) > 0.1, True)

    bt2 = BackTest('C:\\Users\\user\\absolute\\location\\of\\data')
    bt2.import_data(start_date='2019-06-01', end_date='2021-06-01')
    bt2.generate_signals(s2)
    bt2.run_backtest(stoploss=0.10)
    bt2.report()

    # longer period
    bt3 = BackTest('C:\\Users\\user\\absolute\\location\\of\\data')
    bt3.import_data(start_date='2010-06-01', end_date='2021-12-31')
    bt3.generate_signals(s2)
    bt3.run_backtest(stoploss=0.10)
    bt3.report()
