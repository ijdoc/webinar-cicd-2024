from scipy.stats import ks_2samp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wandb.apis.reports.v1 as wr  # For creating reports
import os
import requests
import subprocess
import re


def ecdf(data):
    x = np.sort(data)
    n = len(x)
    y = np.arange(1, n + 1) / n
    return x, y


def plot_ecdf(feature_name, data1, data2, label1="Original", label2="Candidate"):
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
    result = p_value < threshold
    return {"p_value": p_value, "drifted": result}


def run_drift_check(initial_data, new_data, feature_columns, threshold=0.05):
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
            initial_data[feature], new_data[feature], threshold=threshold
        )

    return drift_results


def make_report(entity, project, run_name, threshold, drift_detected, media_keys):
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
                f"Each feature was checked separately for drift using a KS test with "
                f"`p < {threshold}`. `False` means no drift was detected, while "
                "`True` indicates drift was detected."
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
                "Empirical Cumulative Distribution Function (ECDF) plots are shown below to "
                "facilitate review of feature differences."
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


def open_github_issue(issue_title, issue_body, labels=None):
    """
    Opens a new issue in the GitHub repository.
    """

    repo_owner, repo_name = get_github_repo_info()
    token = os.environ.get("GITHUB_TOKEN")

    # GitHub API endpoint for creating issues
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues"

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    data = {"title": issue_title, "body": issue_body, "labels": labels}

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 201:
        issue_url = response.json()["html_url"]
        return issue_url
    else:
        raise RuntimeError(f"Failed to create GitHub issue: {response.content}")
    return None


def get_github_repo_info():
    try:
        # Get the remote origin URL
        remote_url = (
            subprocess.check_output(
                ["git", "config", "--get", "remote.origin.url"],
                stderr=subprocess.STDOUT,
            )
            .decode()
            .strip()
        )

        # Parse the URL to get owner and repo name
        if remote_url.startswith("git@"):
            # SSH URL
            pattern = r"git@github.com:(.+)/(.+)\.git"
        elif remote_url.startswith("https://"):
            # HTTPS URL
            pattern = r"https://github.com/(.+)/(.+)"
        else:
            # Other formats
            return None, None

        match = re.match(pattern, remote_url)
        if match:
            owner = match.group(1)
            repo = match.group(2)
            return owner, repo
        else:
            return None, None
    except Exception as e:
        print(f"Error retrieving GitHub repo info: {e}")
        return None, None
