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

.. autofunction:: ndxtest.indicators.roc
.. autofunction:: ndxtest.indicators.sma
.. autofunction:: ndxtest.indicators.ema
.. autofunction:: ndxtest.indicators.ssma
.. autofunction:: ndxtest.indicators.rsi
.. autofunction:: ndxtest.indicators.macd
.. autofunction:: ndxtest.indicators.bbands
.. autofunction:: ndxtest.indicators.atr
.. autofunction:: ndxtest.indicators.adx

Crossover indicators
--------------------
The following functions can be used to build conditions based on *crossovers*:

.. autofunction:: ndxtest.indicators.static
.. autofunction:: ndxtest.indicators.crossover
.. autofunction:: ndxtest.indicators.intraday_price_crossover

Candlesticks and price action
-----------------------------
The following functions detect certain types of candlesticks or price actions:

.. autofunction:: ndxtest.indicators.price_action
.. autofunction:: ndxtest.indicators.green_candle
.. autofunction:: ndxtest.indicators.red_candle
.. autofunction:: ndxtest.indicators.bullish_pin_bar
.. autofunction:: ndxtest.indicators.bearish_pin_bar
.. autofunction:: ndxtest.indicators.inside_bar
.. autofunction:: ndxtest.indicators.inside_open

utils.py
~~~~~~~~
**`ndxtest.utils`** contains the internally used class :class:`ndxtest.utils.Portfolio` as well as functions helping to
maintain and update the data library.

Class :class:`ndxtest.utils.LibManager`
---------------------------------------
.. autoclass:: ndxtest.utils.LibManager
   :special-members: __init__
   :members:

Class :class:`ndxtest.utils.Portfolio`
--------------------------------------
.. autoclass:: ndxtest.utils.Portfolio
   :special-members: __init__
   :members:

Miscellaneous
-------------
.. autofunction:: ndxtest.utils.connect
.. autofunction:: ndxtest.utils.constituents
.. autofunction:: ndxtest.utils.timeit_decorator

.. toctree::
   :maxdepth: 2
