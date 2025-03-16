from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd

'''def one_hot_encode(X, cols):
    # One-Hot Encoding:
    encoder_one_hot = OneHotEncoder()
    for col in cols:
        #X = encoder_one_hot.fit_transform(X[[col]])
        X[col] = encoder_one_hot.fit_transform(X[col])
    return X'''

def one_hot_encode(X, cols):
    encoder_one_hot = OneHotEncoder(sparse=False)  # Use sparse=False to return a dense array
    for col in cols:
        # Reshape to 2D array and apply one-hot encoding
        X_encoded = encoder_one_hot.fit_transform(X[[col]])  # X[[col]] is 2D (DataFrame)

        # Convert the encoded values into DataFrame and join back with the original DataFrame
        encoded_df = pd.DataFrame(X_encoded, columns=encoder_one_hot.get_feature_names_out([col]))
        X = X.join(encoded_df).drop(columns=[col])  # Drop original column after encoding

    return X


def label_encode(X, cols):
    #Label Encoding:
    label_encoder = LabelEncoder()
    for col in cols:
        #X = label_encoder.fit_transform(X[[col]])
        X[col] = label_encoder.fit_transform(X[col])
    return X

def encode_target(X):
    X = X.astype(int)
    return X