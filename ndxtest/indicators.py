""""""

import numpy as np
import pandas as pd

_DTYPE = 'float32'


# General Indicators #


def roc(arr, n, input_col='close'):
    """Calculates the rate of change (ROC).

    :param pd.DataFrame or pd.Series arr: A pd.DataFrame containing ohlc price data, or a pd.Series containing price data.
    :param int n: The periods for ROC calculation.
    :param str, default='close' input_col: The input column name if arr is a `pd.DataFrame`

    :returns: A pd.Series containing the ROC. Will contain np.nan values.
    :rtype: pd.Series of dtype np.float32.
    """
    if isinstance(arr, pd.DataFrame):
        arr = arr[input_col]
    return arr.pct_change(periods=n).astype(_DTYPE)


def sma(arr, n, input_col='close', full_window=True):
    """Calculates the simple moving average (SMA).

    :param pd.DataFrame or pd.Series arr: A pd.DataFrame containing ohlc price data, or a pd.Series containing price data.
    :param int n: The periods for SMA calculation.
    :param str, default='close' input_col: The input column name if arr is a `pd.DataFrame`
    :param bool, default=True full_window: If True, the output pd.Series will have n-1 trailing np.nan values.

    :returns: A pd.Series containing the SMA. Will contain np.nan values if full_window=True.
    :rtype: pd.Series of dtype np.float32.
    """
    if isinstance(arr, pd.DataFrame):
        arr = arr[input_col]
    return arr.rolling(n, min_periods=n if full_window else 1).mean().astype(_DTYPE)


def ema(arr, n, input_col='close', full_window=True):
    """Calculates the exponential moving average (EMA).

    :param pd.DataFrame or pd.Series arr: A pd.DataFrame containing ohlc price data, or a pd.Series containing price data.
    :param int n: The periods for EMA calculation.
    :param str, default='close' input_col: The input column name if arr is a `pd.DataFrame`.
    :param bool, default=True full_window: If True, the output pd.Series will have n-1 trailing np.nan values.

    :returns: A pd.Series containing the EMA. Will contain np.nan values if full_window=True.
    :rtype: pd.Series of dtype np.float32.
    """
    if isinstance(arr, pd.DataFrame):
        arr = arr[input_col]
    return arr.ewm(span=n, min_periods=n if full_window else 1).mean().astype(_DTYPE)


def ssma(arr, n, input_col='close', full_window=True):
    """Calculates smoothed simple moving average (SSMA).

    :param pd.DataFrame or pd.Series arr: A pd.DataFrame containing ohlc price data, or a pd.Series containing price data.
    :param int n: The periods for SSMA calculation.
    :param str, default='close' input_col: The input column name if arr is a `pd.DataFrame`.
    :param bool, default=True full_window: If True, the output pd.Series will have n-1 trailing np.nan values.

    :returns: A pd.Series containing the SSMA. Will contain np.nan values if full_window=True.
    :rtype: pd.Series of dtype np.float32.
    """
    if isinstance(arr, pd.DataFrame):
        arr = arr[input_col]
    return arr.ewm(alpha=1 / n, min_periods=n if full_window else 1).mean().astype(_DTYPE)


def atr(arr: pd.DataFrame, n=14, full_window=True):
    """Calculates the average true range (ATR) indicator.

    :param pd.DataFrame arr: A pd.DataFrame containing ohlc price data.
    :param int, default=14 n: The periods for the atr calculation. Usually the 14 period atr is used.
    :param bool, default=True full_window: If True, the output pd.Series will have n-1 trailing np.nan values.

    :returns: A pd.Series containing the ATR. Will contain np.nan values if full_window=True.
    :rtype: pd.Series of dtype np.float32.
    """
    if isinstance(arr, pd.DataFrame):
        true_range = np.maximum.reduce([arr.high, arr.close.shift(1)]) - np.minimum.reduce(
            [arr.low, arr.close.shift(1)])
        true_range = pd.Series(data=true_range, index=arr.index).astype(_DTYPE)
        average_true_range = true_range.ewm(alpha=1 / n, min_periods=n if full_window else 1).mean().astype(_DTYPE)
        # relative_true_range = (true_range / average_true_range).astype(_DTYPE)
        return average_true_range
    else:
        raise TypeError("Argument 'arr' must be a pd.DataFrame.")


def adx(arr: pd.DataFrame, n=14, full_window=True):
    """Calculates the average directional index (ADX) indicator. Part of calculating the ADX is calculating the ATR first.

    :param pd.DataFrame arr: A pd.DataFrame containing ohlc price data.
    :param int, default=14 n: The periods for ADX calculation. Usually the 14 period ADX is used.
    :param bool, default=True full_window: If True, the output pd.Series will have n-1 trailing np.nan values.

    :returns: A pd.Series containing the ADX. Will contain np.nan values if full_window=True.
    :rtype: pd.Series of dtype np.float32.
    """
    if isinstance(arr, pd.DataFrame):
        plus_dm = (arr.high - arr.high.shift(1)).astype(_DTYPE)
        minus_dm = (arr.low.shift(1) - arr.low).astype(_DTYPE)
        positive_dm = np.where(np.greater(plus_dm, minus_dm) & np.greater(plus_dm, 0), plus_dm, 0)
        positive_dm = pd.Series(positive_dm, index=arr.index, dtype=_DTYPE)
        negative_dm = np.where(np.greater(minus_dm, plus_dm) & np.greater(minus_dm, 0), minus_dm, 0)
        negative_dm = pd.Series(negative_dm, index=arr.index, dtype=_DTYPE)
        average_true_range = atr(arr, n, full_window=full_window)

        positive_di = (ema(positive_dm, n, full_window=full_window) / average_true_range) * 100
        negative_di = (ema(negative_dm, n, full_window=full_window) / average_true_range) * 100
        directional_index = (abs(positive_di - negative_di) / (positive_di + negative_di)).astype(_DTYPE) * 100
        average_directional_index = \
            directional_index.ewm(alpha=1 / n, min_periods=n if full_window else 1).mean().astype(_DTYPE)
        return average_directional_index
    else:
        raise TypeError("Argument 'arr' must be a pd.DataFrame.")


def rsi(arr, n=14, input_col='close', full_window=True):
    """Calculates the relative strength index (RSI).

    :param pd.DataFrame or pd.Series arr: A pd.DataFrame containing ohlc price data, or a pd.Series containing price data.
    :param int, default=14 n: The periods for the RSI. By convention the 14 period RSI is calculated.
    :param str, default='close' input_col: The input column name if arr is a `pd.DataFrame`.
    :param bool, default=True full_window: If True, the output pd.Series will have n-1 trailing np.nan values.

    :returns: A pd.Series containing the RSI. Will contain np.nan values if full_window=True.
    :rtype: pd.Series of dtype np.float32.

    """
    if isinstance(arr, pd.DataFrame):
        arr = arr[input_col]
    moves = arr.diff()
    up_moves = pd.Series(data=np.where(np.greater_equal(moves, 0), moves, 0), index=moves.index).astype(_DTYPE)
    down_moves = abs(pd.Series(data=np.where(np.greater_equal(0, moves), moves, 0), index=moves.index).astype(_DTYPE))
    rs_index = 100 - (100 / (1 + (up_moves.rolling(n, min_periods=n if full_window else 1).mean() /
                                  down_moves.rolling(n, min_periods=n if full_window else 1).mean()))).astype(_DTYPE)
    return rs_index


def macd(arr, input_col='close', full_window=True):
    """Calculates the moving average convergence divergence (MACD) index. The MACD has fixed periods (12 and 26).

    :param pd.DataFrame or pd.Series arr: A pd.DataFrame containing ohlc price data, or a pd.Series containing price data.
    :param str, default='close' input_col: The input column name if arr is a `pd.DataFrame`.
    :param bool, default=True full_window: If True, the output pd.Series will have 25 trailing np.nan values.

    :returns: Three separate pd.Series: The MACD line, the signal line and the MACD histogram.
    :rtype: a tuple containing three pd.Series of dtype np.float32.
    """
    if isinstance(arr, pd.DataFrame):
        arr = arr[input_col]
    macd_line = arr.ewm(span=12, min_periods=12 if full_window else 1).mean().astype(_DTYPE) - \
                arr.ewm(span=26, min_periods=26 if full_window else 1).mean().astype(_DTYPE)
    signal_line = macd_line.ewm(span=9, min_periods=9 if full_window else 1).mean().astype(_DTYPE)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bbands(arr, n=20, stdl=2, stdu=2, input_col="close", full_window=True):
    """Calculates the a simple moving average (SMA) and its bollinger bands (BB). Usually, the 20 period SMA and its BB's are used.

    :param pd.DataFrame or pd.Series arr: A pd.DataFrame containing ohlc price data, or a pd.Series containing price data.
    :param int n: The periods for SMA calculation.
    :param float stdl: The standard deviation used in calculating the *lower* BB.
    :param float stdu: The standard deviation used in calculating the *upper* BB.
    :param str, default='close' input_col: The input column name if arr is a `pd.DataFrame`.
    :param bool, default=True full_window: If True, the output pd.Series will have n-1 trailing np.nan values.

    :returns: Three separate pd.Series: The SMA, the lower BB and the upper BB.
    :rtype: a tuple containing three pd.Series of dtype np.float32.
    """
    if isinstance(arr, pd.DataFrame):
        arr = arr[input_col]
    ma = arr.rolling(n, min_periods=n if full_window else 1).mean().astype(_DTYPE)
    lbb = (ma - arr.rolling(n, min_periods=n if full_window else 1).std() * stdl).astype(_DTYPE)
    ubb = (ma + arr.rolling(n, min_periods=n if full_window else 1).std() * stdu).astype(_DTYPE)
    return ma, lbb, ubb


# Crossovers #


def static(s, arr):
    """Generates a static level to be used in crossover strategies.

    :param int s: An integer.
    :param pd.DataFrame or pd.Series arr: A pd.DataFrame or pd.Series.

    :returns: A pd.Series that has the index of arr and values of s.
    :rtype: pd.Series of dtype np.int8
    """
    return pd.Series(data=s, index=arr.index, dtype=np.int8)


def crossover(slow: pd.Series or int, fast: pd.Series, direction: str):
    """Checks for bullish or bearish crossovers of a faster-moving series through a slower-moving or static series.

    The parameter `slow` can be either a pd.Series or an integer. If an integer is provided it will be converted to
    a static pd.Series with the index of `fast`.

    Examples of usage:

    - crossover(sma(arr, 50), sma(arr, 20), 'bullish') -> Bullish crossovers of the 20 period SMA through the 50 period SMA.
    - crossover(50, rsi(arr), 'bearish') -> Bearish crossovers whenever the RSI falls below the 50 level.

    :param pd.Series or int slow: A pd.Series or an integer.
    :param pd.Series fast: A pd.Series.
    :param str, direction: Must be either 'bullish' or 'bearish'.

    :returns: A pd.Series. True where bullish or bearish crossovers have occurred, respectively.
    :rtype: pd.Series of dtype bool
    """

    if not isinstance(fast, pd.Series):
        raise TypeError("Argument 'fast' must be of type pd.Series.")

    if not isinstance(slow, pd.Series):
        if isinstance(slow, int):
            slow = pd.Series(data=slow, index=fast.index, dtype=np.int8)
        else:
            raise TypeError("Argument 'slow' must be of type pd.Series or int.")

    if not isinstance(direction, str):
        raise TypeError("Argument 'direction' must be of type str.")

    if direction == 'bullish':
        bullish = True
    elif direction == 'bearish':
        bullish = False
    else:
        raise ValueError("Argument 'direction' must be either 'bullish' or 'bearish'.")

    bearish_crossovers = ((slow.shift(1) < fast.shift(1)) & (fast < slow))
    bullish_crossovers = ((slow.shift(1) > fast.shift(1)) & (fast > slow))
    return bullish_crossovers if bullish else bearish_crossovers


def intraday_price_crossover(arr: pd.DataFrame, series: pd.Series, direction: str):
    """Checks for bullish or bearish intraday-crossovers of the price through a moving series.

    Examples:

    - intraday_price_crossover(arr, sma(arr, 100), 'bearish') -> True where a candle opens above and closes beneath the 100 period SMA.

    :param pd.DataFrame arr: A pd.DataFrame containing ohlc price data.
    :param pd.Series series: A pd.Series, e.g. a moving average.
    :param str, direction: Must be either 'bullish' or 'bearish'.

    :returns: A pd.Series. True where bullish or bearish crossovers have occurred, respectively.
    :rtype: pd.Series of dtype bool
    """

    if not isinstance(arr, pd.DataFrame):
        raise TypeError("Argument 'arr' must be of type pd.DataFrame.")

    if not isinstance(series, pd.Series):
        raise TypeError("Argument 'series' must be of type pd.Series.")

    if not isinstance(direction, str):
        raise TypeError("Argument 'direction' must be of type str.")

    if direction == 'bullish':
        bullish = True
    elif direction == 'bearish':
        bullish = False
    else:
        raise ValueError("Argument 'direction' must be either 'bullish' or 'bearish'.")

    bullish_crossovers = ((series > arr.open) & (arr.close > series))
    bearish_crossovers = ((series < arr.open) & (arr.close < series))

    return bullish_crossovers if bullish else bearish_crossovers


# Candlesticks and Price Action #


def candle(arr, prev='c', ineq='gt', curr='c'):
    """Versatile function to compare ohlc data of the previous and the current candle.

    Examples:

    - candle(arr, prev='c', ineq='gt', curr='c') -> True wherever the previous close was greater than the current close.
    - candle(arr, prev='l', ineq='gt', curr='o') -> True wherever the previous low was greater than the current open. (A gap down candle.)

    :param pd.DataFrame arr: A pd.DataFrame containing ohlc price data.
    :param str prev: Datapoint of the previous candle. Can be either 'o', 'h', 'l' or 'c'.
    :param str ineq: Inequality operator. Can be either 'gt', 'ge', 'lt' or 'le'.
    :param str curr: Datapoint of the current candle. Can be either 'o', 'h', 'l' or 'c'.

    :returns: A pd.Series, True wherever the conditions were met.
    :rtype: pd.Series of dtype bool
    """

    if not isinstance(prev, str) and isinstance(curr, str) and isinstance(ineq, str):
        raise TypeError("Arguments 'prev', 'curr' and 'ineq' must be of type str.")

    if prev not in ['o', 'h', 'l', 'c']:
        raise ValueError("Argument 'prev' must be either 'o', 'h', 'l' or 'c'.")

    if curr not in ['o', 'h', 'l', 'c']:
        raise ValueError("Argument 'curr' must be either 'o', 'h', 'l' or 'c'.")

    if ineq not in ['gt', 'ge', 'lt', 'le']:
        raise ValueError("Argument 'curr' must be either 'gt', 'ge', 'lt' or 'le'.")

    ohlc_map = {'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close'}
    ineq_map = {
        'gt': np.greater,
        'ge': np.greater_equal,
        'lt': np.less,
        'le': np.less_equal}

    return ineq_map[ineq](arr[ohlc_map[prev]].shift(1), arr[ohlc_map[curr]])


def green_candle(arr):
    """Green candle.

    :param pd.DataFrame arr: A pd.DataFrame containing ohlc price data.

    :returns: A pd.Series, True for all candles that closed higher than they opened.
    :rtype: pd.Series of dtype bool
    """

    return arr['close'] > arr['open']


def red_candle(arr):
    """Red candle.

    :param pd.DataFrame arr: A pd.DataFrame containing ohlc price data.

    :returns: A pd.Series, True for all candles that closed lower than they opened.
    :rtype: pd.Series of dtype bool
    """

    return arr['close'] < arr['open']


def bullish_pin_bar(arr, j=3):
    """Detects bullish pin bars.

    Herein, bullish pin bars are defined as follows:

    - The lower wick of the candle must be at least three (j) times longer than the body of the candle.
    - The lower boundary of the candle-body must be situated in the upper third of the candle.
    - The body of the candle must be within the body of the previous candle
    - The lower wick of the pin bar must reach a lower low than the previous candle.

    :param pd.DataFrame arr: A pd.DataFrame containing ohlc price data.
    :param int, default=3 j: The ratio of lower wick length to candle body length.

    :returns: A pd.Series, True where bullish pin bars were detected.
    :rtype: pd.Series of dtype bool
    """

    body_length = abs(arr['close'] - arr['open'])
    lower_wick_length = np.minimum(arr['close'], arr['open']) - arr['low']
    prev_min = np.minimum(arr['close'].shift(1), arr['open'].shift(1))
    prev_max = np.maximum(arr['close'].shift(1), arr['open'].shift(1))
    body_inside_prev_body = (prev_min < arr['open']) & (arr['open'] < prev_max) & \
                              (prev_min < arr['close']) & (arr['close'] < prev_max)
    body_in_upper_third = np.minimum(arr['open'], arr['close']) > (arr['low'] + (2 * arr['high'])) / 3
    wick_lt_prev = arr['low'] < arr['low'].shift(1)
    return (lower_wick_length > j * body_length) & body_in_upper_third & wick_lt_prev & body_inside_prev_body


def bearish_pin_bar(arr, j=3):
    """Detects bearish pin bars.

    Herein, bearish pin bars are defined as follows:

    - The upper wick of the candle must be at least three (j) times longer than the body of the candle.
    - The upper boundary of the candle-body must be situated in the lower third of the candle.
    - The body of the candle must be within the body of the previous candle
    - The upper wick of the pin bar must reach a higher high, compared to the upper wick of the previous candle.

    :param pd.DataFrame arr: A pd.DataFrame containing ohlc price data.
    :param int, default=3 j: The ratio of lower wick length to candle body length.

    :returns: A pd.Series, True where bearish pin bars were detected.
    :rtype: pd.Series of dtype bool
    """

    body_length = abs(arr['close'] - arr['open'])
    upper_wick_length = arr['high'] - np.maximum(arr['close'], arr['open'])
    prev_min = np.minimum(arr['close'].shift(1), arr['open'].shift(1))
    prev_max = np.maximum(arr['close'].shift(1), arr['open'].shift(1))
    body_inside_prev_body = (prev_min < arr['open']) & (arr['open'] < prev_max) & \
                              (prev_min < arr['close']) & (arr['close'] < prev_max)
    body_in_lower_third = np.maximum(arr['open'], arr['close']) < ((2 * arr['low']) + arr['high']) / 3
    wick_gt_prev = (arr['high'] < arr['high'].shift(1)) & (arr['low'] < arr['low'].shift(-1))
    return (upper_wick_length > j * body_length) & body_inside_prev_body & body_in_lower_third & wick_gt_prev


def inside_bar(arr):
    """Detects inside bars."""
    return (arr['high'] < arr['high'].shift(1)) & (arr['low'] > arr['low'].shift(1))


def gap_up_body(arr):
    return np.maximum(arr['close'].shift(1), arr['open'].shift(1)) < arr['open']


def gap_down_body(arr):
    return np.minimum(arr['close'].shift(1), arr['open'].shift(1)) > arr['open']


def gap_up_wick(arr):
    return arr['high'].shift(1) < arr['open']


def gap_down_wick(arr):
    return arr['low'].shift(1) > arr['open']


def inside_open(arr):
    return (np.minimum(arr['close'].shift(1), arr['open'].shift(1)) < arr['open']) & \
           (np.maximum(arr['close'].shift(1), arr['open'].shift(1)) > arr['open'])
