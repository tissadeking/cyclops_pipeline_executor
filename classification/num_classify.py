from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

def pac_model_fun(X_train, y_train):
    model = PassiveAggressiveClassifier()
    model.fit(X_train, y_train)
    return model

def log_model_fun(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def mnb_model_fun(X_train, y_train):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

def mlpc_model_fun(X_train, y_train):
    model = MLPClassifier(max_iter=1000000)
    model.fit(X_train, y_train)
    return model

def svc_model_fun(X_train, y_train):
    model = SVC()
    model.fit(X_train, y_train)
    return model

def dt_model_fun(X_train, y_train):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model

def kn_model_fun(X_train, y_train):
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    return model

def rf_model_fun(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model
