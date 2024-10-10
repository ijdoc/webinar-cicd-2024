import matplotlib.pyplot as plt
import numpy as np


# Function to plot predictions vs. actuals
def plot_predictions_vs_actuals(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.3)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Predictions vs Actuals")
    return plt


def prep_time_series_data(X, y, n_time_steps):
    X_time_series = []
    y_time_series = []
    for i in range(len(X) - n_time_steps):
        X_time_series.append(
            X[i : i + n_time_steps].flatten()
        )  # Flatten N time steps into one vector
        y_time_series.append(
            y[i + n_time_steps]
        )  # Target is the next time step after N steps

    return np.array(X_time_series), np.array(y_time_series)
