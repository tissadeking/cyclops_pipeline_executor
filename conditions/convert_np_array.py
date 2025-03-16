import numpy as np

def convert_np_array_fun(df, target):
    # Create X by excluding the target column
    cols = df.drop(columns=[target]).columns.tolist()
    #print(cols)
    #X = np.asarray(df.drop(columns=[target]).columns.tolist())
    X = np.asarray(df[cols])
    y = np.asarray(df[target])
    return X, y