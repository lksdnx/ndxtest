"""This is the ndxtest package. A newbies python package for backtesting trading strategies on an index level.

This package relies on a specific set of data to be present. The data can be downloaded from:
https://github.com/lksdnx/ndxtest.

For information on the usage of this package please refer to the online documentation.

Modules
_______
the subpackage ndxtest.core contains the following modules:
    backtest.py
        The main module containing BackTest class and the bulk of the code.
    strategy.py
        Contains the Strategy class that is used to generate strategies that can be supplied to the BackTest class
    utils.py
        Contains functions that help to update the data library and the histfile.
the subpackage ndxtest.indicators contains the following modules:
    candlesticks.py
        Contains functions that check for the presence of specific candle types and patterns.
    comparative.py
        Contains functions that rank symbols within the index based on some other supplied indicators.
    crossovers.py
        Contains functions that check for crossovers of e.g. moving averages or the price and a moving average.
    indicators.py
        The main module here. Contains many common indicators such as moving averages, the RSI or the MACD.
    priceaction.py
        Contains functions that check for specific price actions, e.g. gap downs/gap ups.
        May be joined with candlesticks.py in future releases.

Functions
---------
set_data_path(data_path)
    This function sets the DATA_PATH, which has to represent the absolute location of the `data` folder.
    DATA_PATH is a global variable available to all functions within the ndxtest package.

Constants
---------
DATA_PATH
    See above.
"""
