import logging
from sklearn.model_selection import cross_val_score

log_file = 'cross_validation.log'
logging.basicConfig(filename=log_file, level=logging.INFO)


def perform_cross_validation(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    avg_mse = -scores.mean()

    logging.info(f"Cross-validation for {model} completed.")
    logging.info(f"Average MSE: {avg_mse}")
    print(f"Cross-validation for {model} completed. Average MSE: {avg_mse}")

    return avg_mse
