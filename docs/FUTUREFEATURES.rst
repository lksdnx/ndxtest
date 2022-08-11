Future Features and Open Issues
===============================

New Features
------------

In general, the goal of future releases of `ndxtest` will be to generalise the packages capabilities while at the
same time increase code-stability and ease of use. There are 1000 large and small things that could be done.
Below is an incomplete list of them with descending importance.

- Implementing a scoring mechanism for signals.
- Working on :class:`ndxtest.Strategy` to increase its flexibility. The fact that currently all elementary conditions
  are chained with the & operator is actually a limitation. It would be nice to be able to have arbitrary independent sets
  of conditions for e.g. entering long positions.
- Covering the whole codebase with unit-tests.
- Implementing parameter optimization for strategies (this is work in progress and will probably be part of 0.0.2).
- Implementing means to evaluate the immediate price movement that follows a signal without actually running the backtest.
- Improving performance if the possibility occurs.
- Generalizing to other timeframes than daily data. (long term)
- Generalizing to other indices and baskets than the S&P 500. (long term)
- Creating a web, or desktop based GUI for `ndxtest`. (long long term)

Open Issues
-----------

At the moment I am unaware of critical issues but I am sure the code will raise exceptions under unknown
special circumstances. If you use `ndxtest` and you encounter such a situation, please let me know on XYZ.


.. toctree::
   :maxdepth: 2