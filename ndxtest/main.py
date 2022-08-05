import datetime as dt
from ndxtest.backtest import BackTest, Strategy, _VERSION
from ndxtest.priceaction import bullish_pin_bar
from ndxtest.priceaction import gap_down_wick, gap_up_wick
from ndxtest.indicators import sma

if __name__ == "__main__":
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)

    bt = BackTest('C:\\Users\\lukas\\OneDrive\\Desktop\\')
    bt.import_data(start_date='2021-01-01', end_date='2022-01-01', lag=50)
    bt.optimize_strategy(1, {})

    exit()
    bt = BackTest(start_date=dt.datetime(2021, 1, 1),
                  end_date=dt.datetime(2022, 6, 1),
                  lag=dt.timedelta(days=50),
                  runtime_messages=True,
                  date_range_messages=False)
    exit()
    s = Strategy()
    s.add_entry_long_cond(-2, gap_down_wick, False)
    s.add_entry_long_cond(-1, bullish_pin_bar, False)
    s.add_entry_long_cond(0, lambda x: sma(x, 20) > sma(x, 50), False)
    s.add_exit_long_cond(0, gap_up_wick, False)

    bt.generate_signals(s)

    bt.run_backtest(eqc_method='full')
    bt.report()

    print(s)

    # bt.setup_search(pattern=[(-2, gap_down_wick, False),
    #                          (-1, gap_down_wick, False),
    #                          (0, bullish_pin_bar, False),
    #                          (1, gap_up_body, True)])

    # se = "df = pd.DataFrame(data=np.random.randn(10000, 4), columns=['open', 'high', 'low', 'close'])"
    # s = "bbands(df)"
    # n = 1000
    # print(timeit.timeit(stmt=s, setup=se, number=n, globals=globals())/n)
