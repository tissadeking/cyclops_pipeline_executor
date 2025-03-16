from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg as AR
from statsmodels.tsa.api import VAR

#univariate time series forecasting
def ar_fun(X):
    X = pd.Series(X)
    #fit model
    model = AR(X, lags=[1])
    model = model.fit()
    return model
    #print(model_fit.summary())

def arima_fun(X):
    X = pd.Series(X)
    #fit model
    model = ARIMA(X, order=(5,1,0))
    model = model.fit()
    return model

#multivariate time series forecasting
def var_fun(df):
    # Train VAR model
    model = VAR(df)
    model = model.fit()
    return model

