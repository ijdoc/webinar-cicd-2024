from scipy.stats import ks_2samp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wandb.apis.reports.v1 as wr  # For creating reports


def ecdf(data):
    x = np.sort(data)
    n = len(x)
    y = np.arange(1, n + 1) / n
    return x, y


def plot_ecdf(feature_name, data1, data2, label1="Initial", label2="New"):
    x1, y1 = ecdf(data1.dropna())
    x2, y2 = ecdf(data2.dropna())
    plt.figure(figsize=(8, 6))
    plt.step(x1, y1, label=label1, where="post")
    plt.step(x2, y2, label=label2, where="post")
    plt.xlabel(feature_name)
    plt.ylabel("ECDF")
    plt.title(f"ECDF Comparison of {feature_name}")
    plt.legend()
    plt.grid(True)
    return plt


def check_feature_drift(feature_initial, feature_new, threshold=0.05):
    """
    Checks for data drift between two feature distributions using the KS test.

    Args:
        feature_initial (pd.Series): Historical data for the feature.
        feature_new (pd.Series): New data for the feature to compare against the historical data.
        threshold (float): p-value threshold for drift detection.

    Returns:
        bool: True if drift is detected, False otherwise.
    """
    ks_stat, p_value = ks_2samp(feature_initial, feature_new)
    return p_value < threshold


def run_drift_check(initial_data, new_data, feature_columns):
    """
    Runs a data drift check on selected feature columns between initial and new data.

    Args:
        initial_data (Dataframe): The initial dataframe.
        new_data (Dataframe): The new (production) dataframe.
        feature_columns (list): List of feature columns to check for drift.

    Returns:
        dict: Dictionary with drift status for each feature.
    """
    drift_results = {}

    # Check drift for each feature
    for feature in feature_columns:
        drift_results[feature] = check_feature_drift(
            initial_data[feature], new_data[feature]
        )

    return drift_results


def make_report(entity, project, run_name, drift_detected, media_keys):
    if drift_detected:
        description = "Drift detected in the data. Please review the report."
    else:
        description = "No drift detected in the data. The data is consistent."

    # Create a new report
    report = wr.Report(
        project=project,
        title=f"Data Drift Report for {run_name}",
        description=description,
    )

    # Create the report blocks
    blocks = []
    blocks.append(
        wr.MarkdownBlock(
            text=(
                "# Summary of Results\n\n"
                "Each feature is checked separately for drift using a KS test. `False` means no"
                "drift was detected, while `True` indicates drift was detected. If one or more"
                "features show drift, it is recommended that the initial dataset (training) be"
                "replaced with the new dataset (production)."
            )
        )
    )

    # Add summary table
    blocks.append(
        wr.PanelGrid(
            panels=[
                wr.WeavePanelSummaryTable(
                    "drift_results", layout={"x": 0, "y": 0, "w": 24, "h": 8}
                ),
            ],
            runsets=[
                wr.Runset(entity, project, name=run_name).set_filters_with_python_expr(
                    f"Name == '{run_name}'"
                ),
            ],
        )
    )

    blocks.append(
        wr.MarkdownBlock(
            text=(
                "# ECDF Plots\n\n"
                "Empirical Cumulative Distribution Function (ECDF) plots are shown below to"
                "facilitate analysis of feature differences."
            )
        )
    )

    # Add summary table
    blocks.append(
        wr.PanelGrid(
            panels=[
                wr.MediaBrowser(
                    media_keys=media_keys[0],
                    num_columns=1,
                    layout={"x": 0, "y": 0, "w": 12, "h": 8},
                ),
                wr.MediaBrowser(
                    media_keys=media_keys[1],
                    num_columns=1,
                    layout={"x": 12, "y": 0, "w": 12, "h": 8},
                ),
                wr.MediaBrowser(
                    media_keys=media_keys[2],
                    num_columns=1,
                    layout={"x": 0, "y": 8, "w": 12, "h": 8},
                ),
                wr.MediaBrowser(
                    media_keys=media_keys[3],
                    num_columns=1,
                    layout={"x": 12, "y": 8, "w": 12, "h": 8},
                ),
            ],
            runsets=[
                wr.Runset(entity, project, name=run_name).set_filters_with_python_expr(
                    f"Name == '{run_name}'"
                ),
            ],
        )
    )

    # Save the report
    report.blocks = blocks
    report.save()

    return report.url
