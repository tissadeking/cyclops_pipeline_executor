import pandas as pd

#def combine_df_fun(*args):
def combine_df_fun(X_arr):
    #X_arr = list(args)
    X_full = pd.concat(X_arr, ignore_index=True)
    return X_full

