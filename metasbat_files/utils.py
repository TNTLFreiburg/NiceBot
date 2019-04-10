import numpy as np

def remove_columns_with_same_value(df, exclude=('train',)):
    cols_multiple_vals = []
    for col in df.columns:
        try:
            values_set = set(df[col])
            has_multiple_vals = len(values_set) > 1
            if has_multiple_vals:
                all_nans = np.all(np.isnan(values_set))
        except TypeError:
            all_nans = False
            # transform to string in case there are lists
            # since lists not hashable to set
            has_multiple_vals = len(set([str(val) for val in df[col]])) > 1
        cols_multiple_vals.append((has_multiple_vals and (not all_nans)))
    cols_multiple_vals = np.array(cols_multiple_vals)
    excluded_cols = np.array([c in exclude for c in df.columns])
    df = df.iloc[:,(cols_multiple_vals | excluded_cols)]
    return df
