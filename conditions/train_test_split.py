from sklearn.model_selection import train_test_split
import numpy as np

def split_fun(X, y, test_fraction):
    '''# Create X by excluding the target column
    cols = df.drop(columns=[target]).columns.tolist()
    #print(cols)
    #X = np.asarray(df.drop(columns=[target]).columns.tolist())
    X = np.asarray(df[cols])
    y = np.asarray(df[target])
    print(len(X))'''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction, random_state=6)
    return X_train, X_test, y_train, y_test

