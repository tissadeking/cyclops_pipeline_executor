from sklearn.model_selection import train_test_split
import numpy as np

def extract_x_y_fun(df, target):
    # Create X by excluding the target column
    cols = df.drop(columns=[target]).columns.tolist()
    #print(cols)
    #X = np.asarray(df.drop(columns=[target]).columns.tolist())
    X = df[cols]
    y = df[target]
    return X, y

def extract_x_y_text_fun(df, target, X):
    y = df[target]
    return X, y