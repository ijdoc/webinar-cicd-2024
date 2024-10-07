import wandb
from scipy.stats import ks_2samp
import pandas as pd
import sys

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


with wandb.init(
    # mode="disabled",
    project="wandb-webinar-cicd-2024",
    job_type="check-drift",
) as run:

    # Grab the latest training and production dataframes
    train_data = (
        run.use_artifact("training_data:latest").get("training_data").get_dataframe()
    )
    prod_artifact = run.use_artifact("production_data:latest")
    prod_data = prod_artifact.get("production_data").get_dataframe()

    drift_results = run_drift_check(
        train_data, prod_data, ["active_power", "temp", "humidity", "pressure"]
    )

    drift_detected = any(drift_results.values())
    if drift_detected:
        print("Drift detected!")
        # TODO: Log prod data as training data and create a report explaining the drift
        artifact = wandb.Artifact("training_data", type="dataset")
        artifact.add(wandb.Table(dataframe=prod_data), "training_data")
        artifact.description = prod_artifact.description
        run.log_artifact(artifact)
    else:
        print("No drift detected.")

    print(drift_results)

    # Print the drift detection result in a parseable format
    print(f"DRIFT_DETECTED={drift_detected}")
