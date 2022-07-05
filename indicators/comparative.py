import pandas as pd
import numpy as np

DTYPE = 'float32'


def performance_ranking(input_data, benchmark, periods=(5, 21, 55, 89, 144), input_col='close',
                        top_percentile=0.975, bottom_percentile=0.025, top_performers=True):

    if isinstance(periods, int):
        periods = [periods]

    d = {}
    for i, val in enumerate(periods):
        df = pd.DataFrame(data={k: df[input_col].pct_change(periods=val).astype(DTYPE)
                                for k, df in input_data.items()},
                          columns=input_data.keys(),
                          index=benchmark.index)
        d[i] = df.T.fillna(df.mean(axis=1)).T
    df = sum(d.values()).rank(axis=1, pct=True)

    return df > top_percentile if top_performers else df < bottom_percentile
