import pandas as pd
import numpy as np
import operator


def candle(arr, prev='c', curr='c', ineq='gt'):
    """pass"""
    if not isinstance(prev, str) and isinstance(curr, str) and isinstance(ineq, str):
        raise TypeError("The arguments 'prev', 'curr' and 'ineq' must be of type str.")

    if prev not in ['o', 'h', 'l', 'c']:
        raise ValueError("The argument 'prev' must equal to either 'o', 'h', 'l' or 'c'.")

    if curr not in ['o', 'h', 'l', 'c']:
        raise ValueError("The argument 'curr' must equal to either 'o', 'h', 'l' or 'c'.")

    if ineq not in ['gt', 'ge', 'lt', 'le']:
        raise ValueError("The argument 'curr' must equal to either 'gt', 'ge', 'lt' or 'le'.")

    ohlc_map = {'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close'}

    inequality_sign_map = {
        'gt': np.greater,
        'ge': np.greater_equal,
        'lt': np.less,
        'le': np.less_equal}

    return inequality_sign_map[ineq](arr[ohlc_map[prev]].shift(1), arr[ohlc_map[curr]])


def green_candle(arr):
    return arr['close'] > arr['open']


def red_candle(arr):
    return arr['close'] < arr['open']


def bullish_pin_bar(arr, j=3):
    """Detects bullish pin bars. The wick must be at least j times the length of the body."""
    body_length = abs(arr['close'] - arr['open'])
    lower_wick_length = np.minimum(arr['close'], arr['open']) - arr['low']
    prev_min = np.minimum(arr['close'].shift(1), arr['open'].shift(1))
    prev_max = np.maximum(arr['close'].shift(1), arr['open'].shift(1))
    body_inside_prev_candle = (prev_min < arr['open']) & (arr['open'] < prev_max) & \
                              (prev_min < arr['close']) & (arr['close'] < prev_max)
    body_in_upper_third = np.minimum(arr['open'], arr['close']) > (arr['low'] + (2 * arr['high'])) / 3
    wick_lt_neighbors = (arr['low'] < arr['low'].shift(1)) & (arr['low'] < arr['low'].shift(-1))
    return (lower_wick_length > j * body_length) & body_inside_prev_candle & body_in_upper_third & wick_lt_neighbors


def bearish_pin_bar(arr, j=3):
    """Detects bearish pin bars. The wick must be at least j times the length of the body."""
    body_length = abs(arr['close'] - arr['open'])
    upper_wick_length = arr['high'] - np.maximum(arr['close'], arr['open'])
    prev_min = np.minimum(arr['close'].shift(1), arr['open'].shift(1))
    prev_max = np.maximum(arr['close'].shift(1), arr['open'].shift(1))
    body_inside_prev_candle = (prev_min < arr['open']) & (arr['open'] < prev_max) & \
                              (prev_min < arr['close']) & (arr['close'] < prev_max)
    body_in_lower_third = np.maximum(arr['open'], arr['close']) < ((2 * arr['low']) + arr['high']) / 3
    wick_gt_neighbors = (arr['high'] < arr['high'].shift(1)) & (arr['low'] < arr['low'].shift(-1))
    return (upper_wick_length > j * body_length) & body_inside_prev_candle & body_in_lower_third & wick_gt_neighbors


def inside_bar(arr):
    """Detects inside bars."""
    return (arr['high'] < arr['high'].shift(1)) & (arr['low'] > arr['low'].shift(1))


def bullish_pib_pattern(arr):
    """Detects pin bars followed by inside bars. The signal is generated on the inside bar."""
    return bullish_pin_bar(arr).shift(1) & inside_bar(arr)


def bearish_pib_pattern(arr):
    """Detects pin bars followed by inside bars. The signal is generated on the inside bar."""
    return bearish_pin_bar(arr).shift(1) & inside_bar(arr)


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
