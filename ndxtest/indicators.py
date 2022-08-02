"""doc"""
import numpy as np
import pandas as pd

DTYPE = 'float32'


def roc(arr, n=90, input_col='close'):
    """Calculates the rate of change. Performance: avg 0.48 ms per 10000 row df."""
    if isinstance(arr, pd.DataFrame):
        arr = arr[input_col]
    return arr.pct_change(periods=n).astype(DTYPE)


def sma(arr, n, input_col='close', full_window=True):
    """Calculates simple moving averages. Performance: avg 0.59 ms per 10000 row df."""
    if isinstance(arr, pd.DataFrame):
        arr = arr[input_col]
    return arr.rolling(n, min_periods=n if full_window else 1).mean().astype(DTYPE)


def ema(arr, n, input_col='close', full_window=True):
    """Calculates exponential moving averages. Performance: avg 0.27 ms per 10000 row df."""
    if isinstance(arr, pd.DataFrame):
        arr = arr[input_col]
    return arr.ewm(span=n, min_periods=n if full_window else 1).mean().astype(DTYPE)


def ssma(arr, n, input_col='close', full_window=True):
    """Calculates smoothed simple moving averages. Performance: avg 0.30 ms per 10000 row df."""
    if isinstance(arr, pd.DataFrame):
        arr = arr[input_col]
    return arr.ewm(alpha=1 / n, min_periods=n if full_window else 1).mean().astype(DTYPE)


def atr(arr: pd.DataFrame, n=14, full_window=True):
    """Calculates the average true range indicator. The 'out' argument defines the output of the function and
    can be either 'true_range', 'average_true_range' or 'relative_true_range' (default = 'average_true_range').
    Performance: 1.05 ms per 10000 row df."""
    true_range = np.maximum.reduce([arr.high, arr.close.shift(1)]) - np.minimum.reduce([arr.low, arr.close.shift(1)])
    true_range = pd.Series(data=true_range, index=arr.index).astype(DTYPE)
    average_true_range = true_range.ewm(alpha=1 / n, min_periods=n if full_window else 1).mean().astype(DTYPE)
    relative_true_range = (true_range / average_true_range).astype(DTYPE)
    return average_true_range


def adx(arr: pd.DataFrame, n=14, full_window=True):
    """Calculates the average directional index. The 'out' argument defines the output of the function and
    can be either 'directional_index' or 'average_directional_index' (default = 'average_directional_index').
    Performance: 3.90 ms per 10000 row df."""
    if isinstance(arr, pd.DataFrame):
        plus_dm = (arr.high - arr.high.shift(1)).astype(DTYPE)
        minus_dm = (arr.low.shift(1) - arr.low).astype(DTYPE)
        positive_dm = np.where(np.greater(plus_dm, minus_dm) & np.greater(plus_dm, 0), plus_dm, 0)
        positive_dm = pd.Series(positive_dm, index=arr.index, dtype=DTYPE)
        negative_dm = np.where(np.greater(minus_dm, plus_dm) & np.greater(minus_dm, 0), minus_dm, 0)
        negative_dm = pd.Series(negative_dm, index=arr.index, dtype=DTYPE)
        average_true_range = atr(arr, n, full_window=full_window)

        positive_di = (ema(positive_dm, n, full_window=full_window) / average_true_range) * 100
        negative_di = (ema(negative_dm, n, full_window=full_window) / average_true_range) * 100
        directional_index = (abs(positive_di - negative_di) / (positive_di + negative_di)).astype(DTYPE) * 100
        average_directional_index = \
            directional_index.ewm(alpha=1 / n, min_periods=n if full_window else 1).mean().astype(DTYPE)
        return average_directional_index
    else:
        raise TypeError("Argument 'arr' must be a pandas.DataFrame.")


def rsi(arr, n=14, input_col='close', full_window=True):
    """Calculates the relative strength index. Performance: 1.73 ms per 10000 row df."""
    if isinstance(arr, pd.DataFrame):
        arr = arr[input_col]
    moves = arr.diff()
    up_moves = pd.Series(data=np.where(np.greater_equal(moves, 0), moves, 0), index=moves.index).astype(DTYPE)
    down_moves = abs(pd.Series(data=np.where(np.greater_equal(0, moves), moves, 0), index=moves.index).astype(DTYPE))
    rs_index = 100 - (100 / (1 + (up_moves.rolling(n, min_periods=n if full_window else 1).mean() /
                                  down_moves.rolling(n, min_periods=n if full_window else 1).mean()))).astype(DTYPE)
    return rs_index


def macd(arr, input_col='close', full_window=True):
    """Calculates the moving average convergence divergence index. Performance: x.xx ms per 10000 row df."""
    if isinstance(arr, pd.DataFrame):
        arr = arr[input_col]
    macd_line = arr.ewm(span=12, min_periods=12 if full_window else 1).mean().astype(DTYPE) - \
        arr.ewm(span=26, min_periods=26 if full_window else 1).mean().astype(DTYPE)
    signal_line = macd_line.ewm(span=9, min_periods=9 if full_window else 1).mean().astype(DTYPE)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bbands(arr, n=20, stdl=2, stdu=2, input_col="close", full_window=True):
    """Calculates the 20 period sma and its bollinger bands: 'lbb' and 'ubb'. Returns ma, lbb, ubb.
    Performance: 1.94 ms per 10000 row df."""
    if isinstance(arr, pd.DataFrame):
        arr = arr[input_col]
    ma = arr.rolling(n, min_periods=n if full_window else 1).mean().astype(DTYPE)
    lbb = (ma - arr.rolling(n, min_periods=n if full_window else 1).std() * stdl).astype(DTYPE)
    ubb = (ma + arr.rolling(n, min_periods=n if full_window else 1).std() * stdu).astype(DTYPE)
    return ma, lbb, ubb