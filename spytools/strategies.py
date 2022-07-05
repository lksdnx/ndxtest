from indicators.comparative import performance_ranking
from indicators.indicators import roc, sma, bbands, rsi
from indicators.strategy_elements import bullish_crossover, static_level


class Signal:
    def __init__(self):
        pass


def str1(input_data, benchmark, parameters=None):
    """The first 'real' strategy"""
    data = input_data.copy()

    parameters = {} if parameters is None else parameters
    # p1 = parameters['p1'] if 'p1' in parameters else 70
    # p2 = parameters['p2'] if 'p2' in parameters else 20
    # p3 = parameters['p3'] if 'p3' in parameters else 90

    flash_crash = roc(benchmark, 10) < -0.1
    bad_performance = performance_ranking(input_data=input_data, benchmark=benchmark, top_performers=False)
    good_performance = performance_ranking(input_data=input_data, benchmark=benchmark, top_performers=True, top_percentile=0.85)

    for symbol, df in data.items():
        data[symbol]['score'] = 0
        data[symbol]['entry_signals'] = 0
        data[symbol]['exit_signals'] = 0

        high_volume = df.volume > 2 * sma(df.volume, n=20)
        breakout = df.close > bbands(df, stdu=3)[2]
        slowdown_of_momentum = df.close < bbands(df, stdu=1)[1]

        entry_long_where = breakout & high_volume & good_performance[symbol]
        exit_long_where = slowdown_of_momentum

        entry_short_where = flash_crash & bad_performance[symbol]
        exit_short_where = bullish_crossover(static_level(40, df), rsi(df)) | bullish_crossover(sma(benchmark, n=50), benchmark.close)

        # exit_everything_where = bearish_crossover(sma(benchmark, 100), ema(benchmark, 20))

        data[symbol].loc[entry_long_where, 'entry_signals'] = 1
        data[symbol].loc[exit_long_where, 'exit_signals'] = -1

        data[symbol].loc[entry_short_where, 'entry_signals'] = -1
        data[symbol].loc[exit_short_where, 'exit_signals'] = 1

        # data[symbol].loc[exit_everything_where, 'exit_signals'] = -2

    return data
