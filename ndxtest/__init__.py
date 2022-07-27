"""This is the ndxtest package. A newbies python package for backtesting trading strategies on an index level.

This package relies on a specific set of data to be present. The data can be downloaded from:
https://github.com/lksdnx/ndxtest.

For information on the usage of this package please refer to the online documentation.

Modules
_______
ndxtest.core contains the following modules:
    backtest.py
        The main module containing BackTest class and the bulk of the code.
    strategy.py
        Contains the Strategy class that is used to generate strategies that can be supplied to the BackTest class
    utils.py
        Contains functions that help to update the data library and the histfile.
ndxtest.indicators contains the following modules:
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
import os

DATA_PATH = ''


def set_data_path(data_path):
    """This function sets the DATA_PATH, which has to represent the absolute location of the `data` folder.

    After setting the data_path, this function also performs some tests regarding the contents of the `data` folder
    that are necessary for the proper functioning of the ndxtest package.

    Parameters
    ----------
    data_path: str
        The data_path has represent the absolute location of the `data` folder.
    """
    if not isinstance(data_path, str):
        raise TypeError('data_path was not of type str')

    if data_path[-1] == '\\':
        pass
    else:
        data_path += '\\'

    if 'data' not in os.listdir(data_path):
        print("data\\ not found in data_path")
        print("Make sure everything needed is in place and then try setting the data_path again!")
        return None

    if 'lib' not in os.listdir(data_path + 'data\\'):
        print("data\\ found, but lib\\ folder not found in data\\")
        print("Make sure everything needed is in place and then try setting the data_path again!")
        return None

    if '^HIST.xlsx' not in os.listdir(data_path + 'data\\lib\\'):
        print("data\\lib\\ found, but '^HIST.xlsx' is missing.")
        print("Make sure everything needed is in place and then try setting the data_path again!")
        return None

    if '^GSPC.csv' not in os.listdir(data_path + 'data\\lib\\'):
        print("data\\lib\\ found, but '^GSPC.csv' is missing.")
        print("Make sure everything needed is in place and then try setting the data_path again!")
        return None

    global DATA_PATH
    DATA_PATH = data_path
    print(f"Setting the DATA_PATH to {data_path} was successful!")
    return None
