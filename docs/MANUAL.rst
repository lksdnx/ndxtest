User Manual
***********

This is the user manual of the **`ndxtest`** package. See also: https://github.com/lksdnx/ndxtest .

Scope and Limitations
---------------------
The `ndxtest` package seeks to make the backtesting of trading strategies accessible. In its current version (0.0.1)
the package is quite specialized and in many ways limited:

- Backtesting is performed on the basket of stocks included in the S&P 500 index. The necessary data is available in the packages GitHub repository.
- `ndxtest` uses daily price data. Other timeframes are not implemented. This comes with a number of limitations as to what kind of trading strategies can be built with `ndxtest`.
- The trades are entered/exited on the open of the day after signal completion. This is the only mode currently implemented. End of day will be implemented ASAP.
- The codebase is currently not subject to systematic unit-testing. This will be implemented in coming versions.

Future versions of `ndxtest` aim to provide more generalized and thoroughly tested toolsets, while still trying to keep it
simple for users.

Installation
------------

- If you do not already have, install Python version 3.9 or higher from https://www.python.org/downloads/
- If you do not already have, install a code editor of your choice. Google: 'Python IDE'
- Installing `git` from https://git-scm.com/downloads will make it most easy to download the the data needed for `ndxtest`

| Consider first setting up a Python virtual environment as you might not want to
| install `ndxtest` within your main package library. Anyways... go to the python
| console and install `ndxtest` like so:

``pip install ndxtest``

`ndxtest` needs a specific set of data that can be downloaded from the GitHub repository. The package
comes with a set of tools that allows users to maintain and update their own copy to the data.
More on that later, however, new versions of the data will be available every other month on the GitHub repository.

| If you have `git` installed, navigate (``cd``) to your favorite directory in the console.
| Then, paste the following line by line into the console:

| ``git clone --depth 1 --filter=blob:none --sparse https://github.com/lksdnx/ndxtest``
| ``cd ndxtest``
| ``git sparse-checkout set data``

| This will download the folder `data` folder https://github.com/lksdnx/ndxtest .
| You can delete the .git folder after. Keep the data.

Updating the Library
--------------------

| Before we can start building our first strategy with `ndxtest`,
| we first have to make sure that the data we are working with is updated
| and includes the most recent days and weeks of price development.

| `ndxdest` has three modules called `backtest`, `indicators` and `utils`.
| You can read more on them in the technical documentation later.

Create a new .py file in your code editor and

``from ndxtest.utils import LibManager``

``lm = LibManager('C:\\Users\\user\\absolute\\location\\of\\data')``

This will raise errors if the required folder and contents are not found in the path.
The :class:`ndxtest.utils.LibManager` class helps us to update the current library.
As of now though, the process is not fully automated...

The updating process involves three steps:

1. Checking for S&P index announcements on https://www.spglobal.com/spdji/en/indices/equity/sp-500/#news-research
2. Updating `data\\lib\\^HIST.xlsx` (the `histfile`)
3. Running the update. Takes about 5 min., during which you need to sit in front of the screen because

The index announcements given on spglobal.com will include index additions/removals and mergers/spinoffs.
Additionally, name changes for ticker symbols can occur. These are not always announced on spglobal.com.
The classmethods of :class:`ndxtest.utils.LibManager` can be used to update the `histfile`.

Lets say that since the last update 'FOO' replaced 'BAR' in the index, 'BAZ' had a rebranding
and is now trading under ticker symbol 'QUX' and 'JE' acquired (the S&P 500 listed company) 'EZ'
and is now trading under ticker symbol 'JEEZ'. The following will do the job:

| ``lm.histfile_new_entry('add', 'FOO', '2022-06-06')``
| ``lm.histfile_new_entry('remove', 'BAR', '2022-06-06')``
| ``lm.histfile_rename_symbol('BAZ', 'QUX')``
| ``lm.histfile_new_entry('remove', 'JE', '2022-06-23')``
| ``lm.histfile_new_entry('remove', 'EZ', '2022-06-23')``
| ``lm.histfile_new_entry('add', 'JEEZ', '2022-06-23')``

Now we can run the update of the library using :func:`ndxtest.utils.LibManager.update_lib`.
Set the new `new_entries` parameter to the number of new entries that have been added to the `histfile`.
If the last update was more than 3 months ago, set the `period` parameter to '6mo' or '1y', respectively.

``lm.update_lib(period='3mo', new_entries=5)``

The updating function uses the yfinance package: https://pypi.org/project/yfinance/
which provides an easy interface to the finance.yahoo.com API, from which daily price data can be
downloaded for free.

First, a backup `data\\lib_backup_YYYY-MM-DD` is created. Then, `update_lib` appends new records to all `active`
symbols in `data\\lib`. By default, 5 years worth of daily price data will be downloaded if a .csv file is not yet present.
A delay of .3 sec per request prevents the API from limiting user access. This is why the whole process takes some time.
The data provided by finance.yahoo includes stock splits, which are automatically processed.

.. note::
   If `ndxtest.utils.LibManager.update_lib` encounters situations it does not know how to handle it will request
   Y/N user input. For the 5 minutes the update will take you should watch the screen from time to time.

.. note::
   If you have the feeling that something went wrong in the process you can delete the `lib` folder and rename the backup
   folder to `lib`.

.. warning::
   It is recommended to update the library at least once every one to two months, otherwise it becomes laborious.
   Also: finance.yahoo might delist for example acquired companies. The price data then becomes inaccessible for download.

Building a first Strategy
-------------------------

Now that the data is up to date we can build a first strategy. Strategies are built using :class:`ndxtest.utils.Strategy`
as well as some of the indicator functions found in :mod:`ndxtest.indicators`. An instantiated Strategy object has to
be fed with sets of conditions for 4 things:

1. conditions for entering long positions > `ndxtest.backtest.Strategy.add_entry_long_cond`
2. conditions for exiting long positions > `ndxtest.backtest.Strategy.add_exit_long_cond`
3. conditions for entering short positions > `ndxtest.backtest.Strategy.add_entry_short_cond`
4. conditions for exiting short positions > `ndxtest.backtest.Strategy.add_exit_short_cond`

All of the 4 condition sets are internally represented by a list of elementary conditions
that are chained by the & operator.

An elementary condition consists of three parameters: `day`, `condition`, `use_index`,  where...

- `day` is the day on which to check for the condition to be true. -1 is one day before signal completion. 0 is the day of signal completion. Positions are opened on the open of day +1.
- `condition` is the condition. In most cases represented by one or several indicator functions wrapped by a lambda function.
- `use_index` is either ``True`` or ``False``. By default, it is ``False`` and can be omitted. If the parameter is set to ``True`` the strategy will check if the condition is True for the S&P500 index, not for the individual stock.

For a list of indicators currently available refer to the technical documentation of indicators.py. For the first
example strategy here, we will use the relative strength index `rsi` the simple moving average `sma` and the
`crossover` function.

The variable ``x`` refers to a :class:`pandas.DataFrame` containing ohlc data.

| ``from ndxtest.backtest import Strategy``
| ``from ndxtest.indicators import sma, rsi, crossover``
|
| ``s1 = Strategy()``
| ``s1.add_entry_long_cond(0, lambda x: crossover(50, rsi(x), 'bullish'))``
| ``s1.add_entry_long_cond(0, lambda x: x.close > sma(x, 20))``
|
| ``s1.add_exit_long_cond(0, lambda x: crossover(sma(x, 20), x.close, 'bearish'))``

This translated into:
- Buy when the RSI rises above 50 and at the same time the closing price is higher than the 20 period SMA.
- Sell when the price closes beneath the 20 Period SMA.

This simple strategy generates lots of signals and with a universe of 500 stock in an upmarket,
you will fully invested for most of the time.

.. note::
   Take care not to combine any mutually exclusive elementary conditions or you will end up with no signals.

Running the Backtest
--------------------

Lets have a look at the results of ``s1``.

| ``bt = BackTest('C:\\Users\\user\\absolute\\location\\of\\data')``
| ``bt.import_data(start_date='2019-01-01', end_date='2020-12-31', lag=50)``

.. note::
   Upon initialization of a BackTest instance, a timestamped (YYYY-MM-DD_HH-MM-SS) output folder is created
   in the `data\\` directory.


.. note::
   In the example above, `lag=50` tells the BackTest instance to import an additional 50 days of price data
   preceding the `start_date`. This is needed for proper calculation of lagging indicators such as moving
   averages. If your strategy uses a 200 period moving average, you need to increase the lag to 200.

The chosen time period includes the corona crash. Lets see how our strategy performs.

| ``bt.generate_signals(s)``
| ``bt.run_backtest()``
| ``bt.report()``

.. note::
   ``run_backtest()`` has a number of parameters, all of which have default values. For example, you can increase
   the maximum number of positions in your portfolio. Please refer the technical documentation.

The output folder now contains `backtest_report.pdf`, `equity_curve.csv` and `tradelog.csv`. `equity_curve.csv` contains
the ohlc absolute market value development of the Portfolio during the backtest. A plot of the relative development
compared to the index is shown in the report. `tradelog.csv` is a log of all trades taken during the backtest.

The `backtest_report` should look like this:

.. image:: images/report1.png
   :width: 600
   :alt: Alternative text.

We do have a small profit but we dramatically underperformed the index. Before we try to improve on this, lets talk
about a few technical details.

- Some strategies frequently produce a lot of signals on one day, while producing little or no signals on most other days.
  At the moment, a mechanism for ranking the signals so that from lets say 20 signals the *best* ones measured by some metric
  are taken first, is **not implemented**. This will be one of the first new features in upcoming versions of ndxtest.
  At the moment, the signals are taken alphabetically. Lets say the portfolio has only two slots left but there are buy
  signals for 'AAPL', 'META' and 'GOOG'. Positions in 'AAPL' and 'GOOG' but not 'META' will be initiated.
- If a long Position in a given symbol already exists, new entry long signals for this symbol will be ignored and vice versa.
- If an entry short signal for a given symbol occurs but there already exists a long position in this symbol,
  the long position will be closed and *no short position will be opened*. This is a design choice. A more aggressive way
  would be to close the existing long position and at the same time open a short position. The optionality may be implemented
  in future releases.
- The same applies vice versa. A new entry long signal will close an existing short position and not initiate a long position.
- Also read about the `max_trade_duration` and `stoploss` parameters of the `run_backtest()` class method. They both
  represent additional methods to close positions irrespective of the signals provided by the strategy.

Now lets try to improve on ``s1`` and build ``s2``


````

Evaluating trading signals
--------------------------


.. toctree::
   :maxdepth: 2
