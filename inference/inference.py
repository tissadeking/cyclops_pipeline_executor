import pickle

def inference_fun(X_test, filename):
    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    #result = loaded_model.score(X_test, y_test)
    y_pred = loaded_model.predict(X_test)
    return y_pred
    #test_clusters = kmeans.predict(X_test)

def inference_forecast(X_test, filename):
    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    y_pred = loaded_model.predict(start=len(X_test) - 1, end=len(X_test) - len(X_test) / 10, dynamic=False)
    return y_pred

def inference_var_forecast(df, filename):
    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    # Forecast the next 5 time steps
    forecast = loaded_model.forecast(df.values[-loaded_model.k_ar:], steps=5)
    #print(forecast)  # Multi-output prediction
    return forecast