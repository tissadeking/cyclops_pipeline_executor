from sklearn.model_selection import train_test_split
import numpy as np

def split_fun(df, target, test_fraction):
    # Create X by excluding the target column
    X = np.asarray(df.drop(columns=[target]).columns.tolist())
    y = np.asarray(df[target])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction, random_state=6)
    return X_train, X_test, y_train, y_test
