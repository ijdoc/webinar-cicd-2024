import wandb
from utils import run_drift_check, plot_ecdf, make_report, open_github_issue
import matplotlib.pyplot as plt
import pandas as pd

with wandb.init(
    # mode="disabled",
    project="wandb-webinar-cicd-2024",
    job_type="check-drift",
) as run:

    # Grab the latest training and production dataframes
    registered_training_dataset = "jdoc-org/wandb-registry-dataset/training:latest"
    train_artifact = run.use_artifact(registered_training_dataset, type="dataset")
    run.config["train_data"] = train_artifact.name
    try:
        train_data = train_artifact.get("training_data").get_dataframe()
    except:
        # Since the artifact contents likely haven't changed, the name of the
        # logged table will revert to production_data from training_data
        train_data = train_artifact.get("production_data").get_dataframe()

    prod_artifact = run.use_artifact("production_data:latest")
    run.config["prod_data"] = prod_artifact.name
    prod_data = prod_artifact.get("production_data").get_dataframe()

    feature_list = ["active_power", "temp", "humidity", "pressure"]

    threshold = 0.05
    drift_results = run_drift_check(
        train_data, prod_data, feature_list, threshold=threshold
    )

    # Generate and log ECDF plots
    media_keys = []
    for feature in feature_list:
        media_key = f"ECDF/{feature}"
        media_keys.append(media_key)
        plt = plot_ecdf(feature, train_data[feature], prod_data[feature])
        run.log({media_key: wandb.Image(plt)})
        plt.close()

    # Generate and log drift detection results dataframe
    df_drift_results = pd.DataFrame.from_dict(drift_results, orient="index")
    df_drift_results.reset_index(inplace=True)
    df_drift_results.rename(
        columns={"index": "Feature", "p_val": "P-Value", "drifted": "Drift Detected"},
        inplace=True,
    )

    drift_detected = df_drift_results["Drift Detected"].any()
    run.log({"drift_results": wandb.Table(dataframe=df_drift_results)})

    # Create a report explaining the drift (or lack thereof)
    report_url = make_report(
        run.entity, run.project, run.name, threshold, drift_detected, media_keys
    )

    run.config["report_url"] = report_url  # Link report to the run

    # Add report to lineage
    with open("report.txt", "w") as f:
        f.write(report_url)

    art_type = "report"
    art_name = f"{run.name}-drift"
    report_artifact = wandb.Artifact(name=art_name, type=art_type)
    report_artifact.description = report_url
    report_artifact.add_file("report.txt")
    report_artifact = run.log_artifact(report_artifact)

    if drift_detected:
        print("> [!IMPORTANT]")
        print("> Drift detected.\n")
        # Log prod data as training data
        artifact = wandb.Artifact("training_data", type="dataset")
        artifact.add(wandb.Table(dataframe=prod_data), "training_data")
        artifact.description = prod_artifact.description
        artifact = run.log_artifact(artifact).wait()
        # Open a github issue asking for manual review
        issue_title = f"Data drift detected on {train_artifact.name}"
        issue_body = (
            f"Data drift has been detected when comparing the registered training dataset "
            f"with recent production data. Please review the [candidate artifact "
            f"`{artifact.name}`](https://wandb.ai/{run.entity}/{run.project}/artifacts/{artifact.type}/{artifact.name}) "
            f"(originally logged as {artifact.source_name}) and the [generated drift report]({report_url}) "
            f"to determine if the registered training data should be updated.\n\n"
            f"To approve the new candidate after review, link it to the training Dataset Registry"
            f"`{registered_training_dataset}`. Otherwise close this issue."
        )
        issue_url = open_github_issue(issue_title, issue_body, labels=["drift", "data"])
        print(
            f"Production batch `{prod_artifact.source_name}` has been logged "
            f"as candidate `{artifact.name}` to replace training data. "
            f"An [issue]({issue_url}) was also created for manual review:\n"
        )
        print(f"- [Data Drift Issue]({issue_url})")
    else:
        print("> No drift detected.\n")

    print(f"- [W&B Run]({run.url})")
    print(f"- [Full data drift report]({report_url})")

    # Optionally print the drift detection result in a parseable format.
    # Helpful if you want to use this result in a CI/CD pipeline
    # to automatically update the data and/or retrain your model.
    # print(f"DRIFT_DETECTED={drift_detected}")
