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

``git clone --depth 1 --filter=blob:none --sparse https://github.com/lksdnx/ndxtest``
``cd ndxtest``
``git sparse-checkout set data``

| This will download the folder `data` folder https://github.com/lksdnx/ndxtest .
| You can delete the .git folder after. Keep the data.

Maintaining the Updating the Library
------------------------------------

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

``lm.histfile_new_entry('add', 'FOO', '2022-06-06')``
``lm.histfile_new_entry('remove', 'BAR', '2022-06-06')``
``lm.histfile_rename_symbol('BAZ', 'QUX')``
``lm.histfile_new_entry('remove', 'JE', '2022-06-23')``
``lm.histfile_new_entry('remove', 'EZ', '2022-06-23')``
``lm.histfile_new_entry('add', 'JEEZ', '2022-06-23')``

Now that the `histfile` is updated we can run the update of the library.
Set the new `new_entries` parameter to the number of new entries that have been added to the `histfile`.
If the last update was more than 3 months ago. Set the `period` parameter to '6mo' or '1y', respectively.

``lm.update_lib(period='3mo', new_entries=5)``

The updating function uses the yfinance package: https://pypi.org/project/yfinance/
which provides an easy interface to the finance.yahoo.com API.





If weekend or US market closed, the next trading day will be set as start date.
If weekend or US market closed, the next trading day will be set as end date.

.. toctree::
   :maxdepth: 2
