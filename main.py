import logging
from preprocessing import generate_data, feature_scaling
from sklearn.linear_model import LinearRegression
from models import run_models
from parameter_tuining import tune_parameters
from plot import plot_results
from cross_validation import perform_cross_validation
from evaluation_metrics import calculate_mse, calculate_mae
from model_selection import select_best_model
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

log_file = 'main.log'
logging.basicConfig(filename=log_file, level=logging.INFO)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = generate_data()

    X_train_scaled, X_test_scaled = feature_scaling(X_train, X_test)

    lr = LinearRegression()
    svr = SVR()
    dsr = DecisionTreeRegressor()
    rdf = RandomForestRegressor()

    y_pred_lr, y_pred_svr, y_pred_rdf, y_pred_trees = run_models(X_train_scaled, X_test_scaled, y_train, y_test)

    best_params = tune_parameters(X_train_scaled, y_train)

    avg_mse = perform_cross_validation(lr, X_test_scaled, y_test)

    mse = calculate_mse(y_test, y_pred_lr)
    mae = calculate_mae(y_test, y_pred_lr)

    best_model = select_best_model({'Linear Regression': lr, 'SVR': svr,
                                    'Random Forest': rdf, 'Decision Trees': dsr},
                                   X_train_scaled, y_train)

    plot_results(y_pred_lr, y_test)

    logging.info("Project execution completed.")
    print("Project execution completed.")
