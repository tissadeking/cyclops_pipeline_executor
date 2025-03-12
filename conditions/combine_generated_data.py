import numpy as np

def combine_gen_data_fun(X_arr):
    X_full = np.vstack(tuple(X_arr))
    #X_full = pd.concat(X_arr, ignore_index=True)
    return X_full

