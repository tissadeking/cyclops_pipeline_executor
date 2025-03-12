import numpy as np, pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.metrics import r2_score, mean_absolute_error


def get_metrics_reg_fun(X_test, y_test, model):
    # Predict the label of the test data
    y_pred = model.predict(X_test)
    # Evaluate the performance of the model
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return r2, mae

def get_metrics_class_fun(X_test, y_test, model):
    # Predict the label of the test data
    y_pred = model.predict(X_test)
    # Evaluate the performance of the model
    score = metrics.accuracy_score(y_test, y_pred)
    return "%0.3f" % score

def get_metrics_forecast_fun(X_test, y_test, model):
    #y_pred = model.predict(len(X_test)-len(X_test)/2, len(X_test))
    # Forecast the previous 10 data points
    y_pred = model.predict(start=len(X_test) - 1, end=len(X_test) - len(X_test)/10, dynamic=False)    # evaluate forecasts
    #rmse = sqrt(mean_squared_error(y_test, y_pred))
    mse = mean_squared_error(y_test, y_pred)
    return mse
    #print('Test RMSE: %.3f' % rmse)'''

def get_metrics_var_forecast_fun(X_train, X_test, y_test, model):
    # Forecast the next steps (same length as test set)
    lag_order = model.k_ar  # Get lag order
    input_data = X_train.values[-lag_order:]  # Use latest lag_order rows for forecasting
    forecast = model.forecast(input_data, steps=len(X_test))
    # Convert forecast to DataFrame
    forecast_df = pd.DataFrame(forecast, index=X_test.index, columns=X_test.columns)
    # Compute MSE for each column
    mse_per_column = {col: mean_squared_error(X_test[col], forecast_df[col]) for col in X_test.columns}
    # Compute overall MSE (average of all columns)
    overall_mse = np.mean(list(mse_per_column.values()))
    return overall_mse
