from sklearn.svm import OneClassSVM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def one_class_svm_fun(df):
    # Select all numerical columns (automatically handles unknown number of features)
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    # Train One-Class SVM
    oc_svm = OneClassSVM(nu=0.05, kernel="rbf", gamma=0.1)
    df["Anomaly"] = oc_svm.fit_predict(df[numerical_cols])
    print(df["Anomaly"])

    '''# Plot results
    plt.scatter(df["Feature1"], df["Feature2"], c=df["Anomaly"], cmap="coolwarm", edgecolors="k")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Anomaly Detection with One-Class SVM")
    plt.show()'''

    return df["Anomaly"]
