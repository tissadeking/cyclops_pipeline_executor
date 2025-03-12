from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def one_hot_encode(X, cols):
    # One-Hot Encoding:
    encoder_one_hot = OneHotEncoder()
    for col in cols:
        X = encoder_one_hot.fit_transform(X[[col]])
    return X


def label_encode(X, cols):
    #Label Encoding:
    label_encoder = LabelEncoder()
    for col in cols:
        X = label_encoder.fit_transform(X[[col]])
    return X