import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.base import is_classifier
import logging

log_file = 'model_selection.log'
logging.basicConfig(filename=log_file, level=logging.INFO)


def perform_cross_validation(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    avg_mse = -scores.mean()

    logging.info(f"Cross-validation for {model} completed.")
    logging.info(f"Average MSE: {avg_mse}")
    print(f"Cross-validation for {model} completed. Average MSE: {avg_mse}")

    return avg_mse


def select_best_model(models, X_train, y_train, cv=5):
    mse_scores = {}

    for model_name, model in models.items():
        if is_classifier(model):
            # If the model is a classifier, fit a new model using the training data
            model.fit(X_train, y_train)
        elif callable(getattr(model, 'fit', None)):
            # If the model has a 'fit' method, assume it's a fit model
            pass
        else:
            raise ValueError(f"Invalid model type for {model_name}")

        mse = perform_cross_validation(model, X_train, y_train, cv)
        mse_scores[model_name] = mse

    best_model_name = min(mse_scores, key=mse_scores.get)
    best_model = models[best_model_name]

    logging.info(f"Best model selected: {best_model_name}")
    print(f"Best model selected: {best_model_name}")

    return best_model
