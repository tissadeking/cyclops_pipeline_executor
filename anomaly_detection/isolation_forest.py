import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

def isolation_fun(df):
    # Select all numerical columns (automatically handles unknown number of features)
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    # Train Isolation Forest
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    df["Anomaly"] = iso_forest.fit_predict(df[numerical_cols])
    #print(df["Anomaly"])
    # Show anomalies
    anomalies = df[df["Anomaly"] == -1]
    print("Anomalies detected:")
    print(anomalies)
    # Plot results
    '''plt.scatter(df["Feature1"], df["Feature2"], c=df["Anomaly"], cmap="coolwarm", edgecolors="k")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Anomaly Detection with Isolation Forest")
    plt.show()'''

    return df["Anomaly"]

