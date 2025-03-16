import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def autoencode_fun(train_data, test_data):
    n_cols = len(train_data.columns)
    #print(n_cols)
    # Build Autoencoder model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(n_cols,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    #model is trained on only normal data at first
    model.compile(optimizer='adam', loss='mse')
    model.fit(train_data, train_data, epochs=50, batch_size=16, verbose=0)

    # model reconstructs any new data it gets and finds the MSE of the reconstruction
    # Detect anomalies based on reconstruction error
    reconstructed = model.predict(test_data)
    mse = np.mean(np.abs(reconstructed - test_data), axis=1)
    threshold = np.percentile(mse, 95)  # Set anomaly threshold

    # Detect anomalies based on reconstruction error
    '''reconstructed = model.predict(train_data)
    mse = np.mean(np.abs(reconstructed - train_data), axis=1)
    threshold = np.percentile(mse, 95)  # Set anomaly threshold'''
    # Identify anomalies
    train_data["Anomaly"] = mse > threshold
    # Show anomalies
    anomalies = train_data[train_data["Anomaly"] == True]
    print("Anomalies detected:")
    print(anomalies)

    # Plot results
    plt.hist(mse, bins=50, color='blue')
    plt.axvline(threshold, color='red', linestyle='dashed', linewidth=2, label="Anomaly Threshold")
    plt.legend()
    plt.xlabel("Reconstruction Error (MSE)")
    plt.ylabel("Frequency (Number of Samples)")
    plt.title("Reconstruction Error for Anomaly Detection")
    plt.show()

