import pickle, numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def inference_fun(X_test, filename):
    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    #result = loaded_model.score(X_test, y_test)
    y_pred = loaded_model.predict(X_test)
    return y_pred
    #test_clusters = kmeans.predict(X_test)

def inference_forecast_arima(X_test, filename):
    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    #predict with sample
    #y_pred = loaded_model.predict(start=len(X_test) - len(X_test) / 10, end=len(X_test) - 1)
    #predict future data
    y_pred = loaded_model.get_forecast(steps=10).predicted_mean  # Forecasts 10 future steps
    return y_pred

def inference_forecast_ar(X_test, filename):
    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    #predict with sample
    #y_pred = loaded_model.predict(start=len(X_test) - len(X_test) / 10, end=len(X_test) - 1)
    #predict future data
    y_pred = loaded_model.forecast(steps=10)  # Forecasts 10 future steps
    return y_pred

def inference_var_forecast(df, filename):
    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    # Forecast the next 5 time steps
    forecast = loaded_model.forecast(df.values[-loaded_model.k_ar:], steps=5)
    #print(forecast)  # Multi-output prediction
    return forecast


def inference_dbscan(new_data, filename):
    # load the model from disk
    dbscan_model = pickle.load(open(filename, 'rb'))
    # Get the core samples and labels
    core_samples = dbscan_model.components_
    core_labels = dbscan_model.labels_
    # Reshape new data for kneighbors (DBSCAN expects 2D array)
    new_data = np.array(new_data).reshape(-1, 1)
    # Calculate the Euclidean distance from each new data point to the core points
    distances = euclidean_distances(new_data, core_samples)

    # Assign labels based on the nearest core points
    # If the minimum distance is less than epsilon, assign the label of the closest core point
    labels = []
    for dist in distances:
        min_dist_index = np.argmin(dist)
        if dist[min_dist_index] < dbscan_model.eps:  # If within epsilon range
            labels.append(core_labels[min_dist_index])
        else:
            labels.append(-1)  # Mark as noise if not close to any core point

    return labels

def inference_km(new_data, filename):
    new_data = np.array(new_data).reshape(-1, 1)
    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    # result = loaded_model.score(X_test, y_test)
    y_pred = loaded_model.predict(new_data)
    return y_pred
    # test_clusters = kmeans.predict(X_test)