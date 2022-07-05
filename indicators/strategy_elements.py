import pandas as pd
import numpy as np


def bullish_crossover(slow_series, fast_series):
    """Generates Signals based on crossovers of two moving series or one moving series and a static level.
    Assign static levels to the slow_series argument if using a static level."""
    bullish_crossings = ((slow_series.shift(1) > fast_series.shift(1)) & (fast_series > slow_series))
    crossovers = pd.Series(data=False, index=slow_series.index)
    crossovers.loc[bullish_crossings] = True
    return crossovers


def bearish_crossover(slow_series, fast_series):
    """Generates Signals based on crossovers of two moving series or one moving series and a static level.
    Assign static levels to the slow_series argument if using a static level."""
    bearish_crossings = ((slow_series.shift(1) < fast_series.shift(1)) & (fast_series < slow_series))
    crossovers = pd.Series(data=False, index=slow_series.index)
    crossovers.loc[bearish_crossings] = True
    return crossovers


def static_level(static, arr):
    """Generates a static level to be used in crossover strategies."""
    return pd.Series(data=static, index=arr.index, dtype=np.int8)


def intraday_price_crossover(price_df, series: pd.Series, bullish_signal=1, bearish_signal=-1):
    """docstring!"""
    bullish_crossings = ((series > price_df.open) & (price_df.close > series))
    bearish_crossings = ((series < price_df.open) & (price_df.close < series))
    crossovers = pd.Series(data=0, index=series.index, dtype=np.int8)
    crossovers.loc[bullish_crossings] = bullish_signal
    crossovers.loc[bearish_crossings] = bearish_signal
    return crossovers
