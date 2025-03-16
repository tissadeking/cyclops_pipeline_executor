from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from keras.models import Sequential
from keras.layers import Dense

def linear_model_fun(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def lasso_model_fun(X_train, y_train):
    model = Lasso(alpha=0.1, max_iter=100000)
    model.fit(X_train, y_train)
    return model

def ridge_model_fun(X_train, y_train):
    model = Ridge(alpha=0.1)
    model.fit(X_train, y_train)
    return model

def svr_model_fun(X_train, y_train):
    model = SVR()
    model.fit(X_train, y_train)
    return model

def dt_reg_model_fun(X_train, y_train):
    model = DecisionTreeRegressor(random_state=0)
    model.fit(X_train, y_train)
    return model

def nn_model_fun(X_train, y_train):
    dim = len(X_train.columns)
    # create ANN model
    model = Sequential()
    # Defining the Input layer and FIRST hidden layer, both are same!
    model.add(Dense(units=5, input_dim=dim, kernel_initializer='normal', activation='relu'))
    # Defining the Second layer of the model
    # after the first layer we don't have to specify input_dim as keras configure it automatically
    model.add(Dense(units=5, kernel_initializer='normal', activation='tanh'))
    # The output neuron is a single fully connected node
    # Since we will be predicting a single number
    model.add(Dense(1, kernel_initializer='normal'))
    # Compiling the model
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    # Fitting the ANN to the Training set
    model.fit(X_train, y_train, batch_size=5, epochs=5, verbose=0)
    return model

def rf_reg_model_fun(X_train, y_train):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model

