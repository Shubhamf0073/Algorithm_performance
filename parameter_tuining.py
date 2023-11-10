from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor


def tune_parameters(X_train, y_train):
    param_grid = {
        "n_estimators": [10, 50, 100],
        "max_depth": [None, 3, 5],
        "min_samples_split": [2, 5, 10]
    }

    rdf = RandomForestRegressor()
    grid_search = GridSearchCV(rdf, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    print(best_params)

    return best_params
