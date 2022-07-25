"""This module contains the Strategy class, which is used to build strategies for the BackTest class in backtest.py.

Imports
-------
pandas as pd

Classes
-------
Strategy:
    The main class of this module. Please refer to the docstring of the class itself for more information.

Functions
---------
deprecated_strategy_function:
    A deprecated function shell that was used as a signal generation function before the Strategy class was there.
    Will be removed in future versions.

For more information please refer to the docstrings for Strategy as well as its methods.
"""
import pandas as pd


class Strategy:
    """Class to build strategies that are given as a parameter to the `generate_signals` method of a BackTest instance.

    Attributes
    ----------
    data: dict, default=None
        A dictionary containing price data like so: {'AAPL': pd.Dataframe, 'AMNZ': pd.Dataframe, ...}
    index: pd.Dataframe, default=None
        A a pd.Dataframe containing price data of the reference index.
    entry_long_conditions: list, default=[]
    exit_long_conditions: list, default=[]
    entry_short_conditions: list, default=[]
    exit_short_conditions: list, default=[]

    Methods
    -------
    add_entry_long_cond(day, condition, use_index=False):
        Adding an entry condition for long positions.
    add_exit_long_cond(day, condition, use_index=False):
        Adding an exit condition for long positions.
    add_entry_short_cond(day, condition, use_index=False):
        Adding an entry condition for short positions.
    add_exit_short_cond(day, condition, use_index=False):
        Adding an entry condition for short positions.
    generate_signals():
        Searches in data for signals based on all entry and exit conditions provided to this strategy.

    For information on how to use this class please refer to the documentation and the class method docstrings.
    """

    def __init__(self, data=None, index=None):
        """Initializes an instance of the Strategy class.

        Parameters
        ----------
        data: Nonetype, default=None
            When an instance of the Strategy class is passed to the `generate_signals` method of a BackTest instance,
            data will be set to a dictionary containing price data like so: {'AAPL': pd.Dataframe, ...}.

        index: Nonetype, default=None
            When an instance of the Strategy class is passed to the `generate_signals` method of a BackTest instance,
            index will be set to a pd.Dataframe containing price data for the reference index.

        For information on how to use the Strategy class please refer to the documentation.
        """
        self.data = data
        self.index = index
        # conditions for entering and exiting long positions
        self.entry_long_conditions = []
        self.exit_long_conditions = []
        # conditions for entering and exiting short positions
        self.entry_short_conditions = []
        self.exit_short_conditions = []

    def add_entry_long_cond(self, day, condition, use_index=False):
        """Adding an entry condition for long positions.

        Parameters
        ----------
             day: int
                Day (relative to day 0 of signal completion) on which to check for the condition to be fulfilled.
             condition: function(pd.Dataframe | pd.Series)
                A function that takes a pd.Dataframe or pd.Series and checks for each row whether conditions
                (defined by the function) are met or not. Returns a boolean pd.Series with the same index.
             use_index: bool, default=True
                If True, returns a boolean pd.Series based on the symbol df (e.g. 'AMZN').
                If False, returns a boolean pd.Series based on the reference index df.

        For information on how conditions can be structured please refer to the documentation.
        """
        self.entry_long_conditions.append((day, condition, use_index))

    def add_exit_long_cond(self, day, condition, use_index=False):
        """Adding an exit condition for long positions.

        Parameters
        ----------
             day: int
                Day (relative to day 0 of signal completion) on which to check for the condition to be fulfilled.
             condition: function(pd.Dataframe | pd.Series)
                A function that takes a pd.Dataframe or pd.Series and checks for each row whether conditions
                (defined by the function) are met or not. Returns a boolean pd.Series with the same index.
             use_index: bool, default=True
                If True, returns a boolean pd.Series based on the symbol df (e.g. 'AMZN').
                If False, returns a boolean pd.Series based on the reference index df.

        For information on how conditions can be structured please refer to the documentation.
        """
        self.exit_long_conditions.append((day, condition, use_index))

    def add_entry_short_cond(self, day, condition, use_index=False):
        """Adding an entry condition for short positions.

        Parameters
        ----------
        day: int
            Day (relative to day 0 of signal completion) on which to check for the condition to be fulfilled.
        condition: function(pd.Dataframe | pd.Series)
            A function that takes a pd.Dataframe or pd.Series and checks for each row whether conditions
            (defined by the function) are met or not. Returns a boolean pd.Series with the same index.
        use_index: bool, default=True
            If True, returns a boolean pd.Series based on the symbol df (e.g. 'AMZN').
            If False, returns a boolean pd.Series based on the reference index df.

        Returns
        -------
        None

        For information on how conditions can be structured please refer to the documentation.
        """
        self.entry_short_conditions.append((day, condition, use_index))

    def add_exit_short_cond(self, day, condition, use_index=False):
        """Adding an exit condition for short positions.

        Parameters
        ----------
        day: int
            Day (relative to day 0 of signal completion) on which to check for the condition to be fulfilled.
        condition: function(pd.Dataframe | pd.Series)
            A function that takes a pd.Dataframe or pd.Series and checks for each row whether conditions
            (defined by the function) are met or not. Returns a boolean pd.Series with the same index.
        use_index: bool, default=True
            If True, returns a boolean pd.Series based on the symbol df (e.g. 'AMZN').
            If False, returns a boolean pd.Series based on the reference index df.

        Returns
        -------
        None

        For information on how conditions can be structured please refer to the documentation.
        """
        self.exit_short_conditions.append((day, condition, use_index))

    def generate_signals(self):
        """Searches in data for signals based on all entry and exit conditions provided to this strategy.

        Discovered signals are annotated as columns 'entry_signals' and 'exit_signals' to each symbol df.
        In the 'entry_signals' column: 0 = no signal (do nothing), 1 = enter long, -1 = enter short.
        In the 'exit_signals' column: 0 = no signal (do nothing), -1 = exit long, 1 = exit short.

        Parameters
        ----------
        No Parameters.

        Returns
        -------
        dict
            A dictionary containing price data like so: {'AAPL': pd.Dataframe, 'AMNZ': pd.Dataframe, ...}, where each df
            is annotated with a 'score', 'entry_signals' and 'exit_signals' column. The 'score' column currently
            has no function (2022-07-25).
            In the 'entry_signals' column: 0 = no signal (do nothing), 1 = enter long, -1 = enter short.
            In the 'exit_signals' column: 0 = no signal (do nothing), -1 = exit long, 1 = exit short.

        For additional information please refer to the documentation.
        """

        if self.data is None or self.index is None:
            print('No data has been passed to this Strategy instance as self.data and/or self.index is None')
            return None

        for symbol, df in self.data.items():  # scanning for entry and exit conditions in data

            self.data[symbol]['score'] = 0
            self.data[symbol]['entry_signals'] = 0
            self.data[symbol]['exit_signals'] = 0

            # intermediate 'index_df' contains the same records as df
            index_df = self.index.loc[df.index.intersection(self.index.index)]

            # entry_long_signals...
            entry_long = pd.DataFrame()
            for i, condition in enumerate(self.entry_long_conditions):
                days, func, use_index = condition
                if use_index:
                    entry_long[i] = func(index_df).shift(-days)
                else:
                    entry_long[i] = func(df).shift(-days)

            entry_long_signals = entry_long.loc[entry_long.T.all()].index.values
            self.data[symbol].loc[entry_long_signals, 'entry_signals'] = 1

            # exit_long_signals...
            exit_long = pd.DataFrame()
            for i, condition in enumerate(self.exit_long_conditions):
                days, func, use_index = condition
                if use_index:
                    exit_long[i] = func(index_df).shift(-days)
                else:
                    exit_long[i] = func(df).shift(-days)

            exit_long_conditions = exit_long.loc[exit_long.T.all()].index.values
            self.data[symbol].loc[exit_long_conditions, 'exit_signals'] = -1

            # entry_short_signals...
            entry_short = pd.DataFrame()
            for i, condition in enumerate(self.entry_short_conditions):
                days, func, use_index = condition
                if use_index:
                    entry_short[i] = func(index_df).shift(-days)
                else:
                    entry_short[i] = func(df).shift(-days)

            entry_short_signals = entry_short.loc[entry_short.T.all()].index.values
            self.data[symbol].loc[entry_short_signals, 'entry_signals'] = -1

            # exit_short_signals...
            exit_short = pd.DataFrame()
            for i, condition in enumerate(self.exit_short_conditions):
                days, func, use_index = condition
                if use_index:
                    exit_short[i] = func(index_df).shift(-days)
                else:
                    exit_short[i] = func(df).shift(-days)

            exit_short_conditions = exit_short.loc[exit_short.T.all()].index.values
            self.data[symbol].loc[exit_short_conditions, 'exit_signals'] = 1

        return self.data


def deprecated_strategy_function(input_data=None):  # ,benchmark=None, parameters=None):
    """This is deprecated."""
    pass
    data = input_data.copy()

    # parameters = {} if parameters is None else parameters
    # p1 = parameters['p1'] if 'p1' in parameters else 70
    # p2 = parameters['p2'] if 'p2' in parameters else 20
    # p3 = parameters['p3'] if 'p3' in parameters else 90

    for symbol, df in data.items():

        entry_long_where = 0
        exit_long_where = 0
        entry_short_where = 0
        exit_short_where = 0
        exit_everything_where = 0
        exit_everything_where += 1

        data[symbol].loc[entry_long_where, 'entry_signals'] = 1
        data[symbol].loc[exit_long_where, 'exit_signals'] = -1
        data[symbol].loc[entry_short_where, 'entry_signals'] = -1
        data[symbol].loc[exit_short_where, 'exit_signals'] = 1

    return data
