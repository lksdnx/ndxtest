"""for development"""
import datetime as dt
from ndxtest.utils import connect
from ndxtest.backtest import BackTest, Strategy
from ndxtest.indicators import crossover, rsi, sma


if __name__ == "__main__":
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)

    # s1
    s = Strategy()
    s.add_entry_long_cond(-1, lambda x: crossover(50, rsi(x), 'bullish'))
    s.add_entry_long_cond(-1, lambda x: x.close > sma(x, 20))
    s.add_exit_long_cond(-1, lambda x: crossover(sma(x, 20), x.close, 'bearish'))

    bt = BackTest('C:\\Users\\lukas\\OneDrive\\Desktop\\')
    bt.import_data(start_date='2019-01-01', end_date='2020-12-31', lag=50)
    # bt.eval_signals(s)
    bt.generate_signals(s)
    bt.run_backtest()
    bt.report()
