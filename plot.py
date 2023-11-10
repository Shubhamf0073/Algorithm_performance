import matplotlib.pyplot as plt


def plot_results(y_pred, y_test):
    plt.scatter(y_pred, y_test)
    plt.plot([0, 1], [0, 1], color="red", linestyle="--")
    plt.xlabel("Predicted Values")
    plt.ylabel("Actual Values")
    plt.title("Predicted vs Actual values")
    plt.show()
