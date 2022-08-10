"""for development"""
import datetime as dt
from ndxtest.utils import LibManager
from ndxtest.backtest import BackTest, Strategy
from ndxtest.indicators import crossover, rsi, sma, bullish_pin_bar, price_action, green_candle, red_candle


if __name__ == "__main__":
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)

    # lm = LibManager('C:\\Users\\lukas\\OneDrive\\Desktop')

    # s1
    # s = Strategy()
    # s.add_entry_long_cond(0, lambda x: crossover(50, rsi(x), 'bullish'))
    # s.add_entry_long_cond(0, lambda x: x.close > sma(x, 20))
    # s.add_exit_long_cond(0, lambda x: crossover(sma(x, 20), x.close, 'bearish'))

    # bt = BackTest('C:\\Users\\lukas\\OneDrive\\Desktop\\')
    # bt.import_data(start_date='2019-01-01', end_date='2020-12-31', lag=50)
    # bt.generate_signals(s)
    # bt.run_backtest()
    # bt.report()

    # s2
    s2 = Strategy()
    # entry long if...
    s2.add_entry_long_cond(-1, lambda x: price_action(x, 'c', 'lt', 'o'))
    s2.add_entry_long_cond(-1, green_candle)
    s2.add_entry_long_cond(0, lambda x: rsi(x) > 80)
    s2.add_entry_long_cond(0, lambda x: x.close > sma(x, 50), True)
    # exit long if...
    s2.add_exit_long_cond(0, lambda x: crossover(60, rsi(x), 'bearish'))
    # enter short if...
    s2.add_entry_short_cond(-3, red_candle)
    s2.add_entry_short_cond(-2, red_candle)
    s2.add_entry_short_cond(-1, red_candle)
    s2.add_entry_short_cond(0, lambda x: x.close < sma(x, 50), True)
    # exit short if...
    s2.add_exit_short_cond(-1, green_candle)
    s2.add_exit_short_cond(0, green_candle)

    bt2 = BackTest('C:\\Users\\lukas\\OneDrive\\Desktop\\')
    bt2.import_data(start_date='2019-01-01', end_date='2020-12-31', lag=100)
    bt2.generate_signals(s2)
    bt2.run_backtest()
    bt2.report()
