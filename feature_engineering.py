import logging
from sklearn.preprocessing import StandardScaler

log_file = 'feature_engineering.log'
logging.basicConfig(filename=log_file, level=logging.INFO)


def feature_scaling(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Add log messages
    logging.info("Feature scaling completed.")
    print("Feature scaling completed successfully.")

    return X_train_scaled, X_test_scaled
