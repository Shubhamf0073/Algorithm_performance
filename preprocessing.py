import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

log_file = 'data_preprocessing.log'
logging.basicConfig(filename=log_file, level=logging.INFO)


def generate_data():
    features = ["feature1", "feature2", "feature3", "feature4", "feature5"]
    X = np.random.rand(12000, 5)
    y = 2 * X[:, 0] + 3 * X[:, 1] + 4 * X[:, 2] + np.random.rand(12000)
    y += np.random.normal(0, 0.1, 12000)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    logging.info("Data generation completed.")
    print("Data generated successfully.")

    return X_train, X_test, y_train, y_test


def feature_scaling(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logging.info("Feature scaling completed.")
    print("Feature scaling completed successfully.")

    return X_train_scaled, X_test_scaled
