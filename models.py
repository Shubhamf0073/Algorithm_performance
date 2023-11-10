from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


def run_models(X_train, X_test, y_train, y_test):
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    # SVR
    svr = SVR(kernel="rbf")
    svr.fit(X_train, y_train)
    y_pred_svr = svr.predict(X_test)

    # Decision Tree Regression
    dsr = DecisionTreeRegressor()
    dsr.fit(X_train, y_train)
    y_pred_trees = dsr.predict(X_test)

    # Random Forest Regression
    rdf = RandomForestRegressor(n_estimators=100, min_samples_split=10, max_depth=None)
    rdf.fit(X_train, y_train)
    y_pred_rdf = rdf.predict(X_test)

    return y_pred_lr, y_pred_svr, y_pred_rdf, y_pred_trees
