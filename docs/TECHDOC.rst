Technical Documentation
***********************

The ndxtest package contains the following modules:

* backtest.py
* indicators.py
* utils.py

backtest.py
~~~~~~~~~~~

.. automodule:: ndxtest.backtest

Class :class:`ndxtest.backtest.BackTest`
----------------------------------------

.. autoclass:: ndxtest.backtest.BackTest
   :special-members: __init__
   :members:

Class :class:`ndxtest.backtest.Strategy`
----------------------------------------

.. autoclass:: ndxtest.backtest.Strategy
   :special-members: __init__
   :members:

indicators.py
~~~~~~~~~~~~~

**`ndxtest.indicators`** contains indicators that can be used to build trading strategies.

General indicators
------------------
The following functions are general indicators:

.. automodule:: ndxtest.indicators
   :members: roc, sma, ema, ssma, atr, adx, rsi, macd, bbands

Crossover indicators
--------------------
The following functions can be used to build conditions based on *crossovers*:

.. automodule:: ndxtest.indicators
   :members: static, crossover, intraday_price_crossover

Candlesticks and price action
-----------------------------
The following functions detect certain types of candlesticks or price actions:

.. automodule:: ndxtest.indicators
   :members: candle, green_candle, red_candle, bullish_pin_bar, bearish_pin_bar

utils.py
~~~~~~~~

.. automodule:: ndxtest.utils

.. toctree::
   :maxdepth: 2
