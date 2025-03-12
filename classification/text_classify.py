from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


#PASSIVE AGGRESSIVE CLASSIFIER
def pac_model_text(X_train, y_train):
    #Create a pipeline of Tfidf Vectorizer and Passive Aggressive Classifier
    pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
                        ('pamodel', PassiveAggressiveClassifier())])
    #Train the model with the train data
    pipeline.fit(X_train, y_train)
    return pipeline

#LOGISTIC REGRESSION MODEL
def log_model_text(X_train, y_train):
    pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
                        ('logreg', LogisticRegression())])
    #Train the model with the train data
    pipeline.fit(X_train, y_train)
    return pipeline

def mnb_model_text(X_train, y_train):
    pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
                          ('Mnb', MultinomialNB())])
    # Train the model with the train data
    pipeline.fit(X_train, y_train)
    return pipeline

def mlpc_model_text(X_train, y_train):
    #Create a pipeline of Tfidf Vectorizer and Passive Aggressive Classifier
    pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
                              ('MLPC', MLPClassifier(max_iter=1000000))])
    #Train the model with the train data
    pipeline.fit(X_train, y_train)
    return pipeline

def svc_model_text(X_train, y_train):
    pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
                              ('SVC', SVC(max_iter=1000000))])
    #Train the model with the train data
    pipeline.fit(X_train, y_train)
    return pipeline

def dt_model_text(X_train, y_train):
    pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
                              ('DT', DecisionTreeClassifier())])
    # Train the model with the train data
    pipeline.fit(X_train, y_train)
    return pipeline

def kn_model_text(X_train, y_train):
    #Create a pipeline of Tfidf Vectorizer and Passive Aggressive Classifier
    pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
                              ('KN', KNeighborsClassifier())])
    #Train the model with the train data
    pipeline.fit(X_train, y_train)
    return pipeline

def rf_model_text(X_train, y_train):
    pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
                              ('RF', RandomForestClassifier())])
    #Train the model with the train data
    pipeline.fit(X_train, y_train)
    return pipeline


