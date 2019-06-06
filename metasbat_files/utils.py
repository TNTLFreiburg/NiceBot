import numpy as np


def expand_row_with_multiple_values(df, column):
    new_df = df
    for row in df.index:
        for column in df.iloc[row]:
            tmp = np.size(column)
            if tmp > 1:
                new_df
