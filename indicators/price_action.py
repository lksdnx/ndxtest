import numpy as np
import pandas as pd


def green_candle(arr):
    return arr['close'] > arr['open']


def red_candle(arr):
    return arr['close'] < arr['open']


def close_gt_prev_close(arr):
    return arr['close'].shift(1) < arr['close']


def close_lt_prev_close(arr):
    return arr['close'].shift(1) > arr['close']


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

