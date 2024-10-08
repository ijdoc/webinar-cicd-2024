import wandb
from utils import run_drift_check, plot_ecdf, make_report
import matplotlib.pyplot as plt
import pandas as pd

with wandb.init(
    # mode="disabled",
    project="wandb-webinar-cicd-2024",
    job_type="check-drift",
) as run:

    # Grab the latest training and production dataframes
    train_artifact = run.use_artifact("jdoc-org/wandb-registry-dataset/training:latest")
    run.config["train_data"] = train_artifact.source_name
    train_data = train_artifact.get("training_data").get_dataframe()

    prod_artifact = run.use_artifact("production_data:latest")
    run.config["prod_data"] = prod_artifact.source_name
    prod_data = prod_artifact.get("production_data").get_dataframe()

    feature_list = ["active_power", "temp", "humidity", "pressure"]

    drift_results = run_drift_check(train_data, prod_data, feature_list)

    # Generate and log ECDF plots
    media_keys = []
    for feature in feature_list:
        media_key = f"ECDF/{feature}"
        media_keys.append(media_key)
        plt = plot_ecdf(feature, train_data[feature], prod_data[feature])
        run.log({media_key: wandb.Image(plt)})
        plt.close()

    # Generate and log drift detection results table
    drift_results_table = pd.DataFrame(
        list(drift_results.items()), columns=["Feature", "Drift Detected"]
    )
    run.log({"drift_results": wandb.Table(dataframe=drift_results_table)})

    drift_detected = any(drift_results.values())
    if drift_detected:
        print("Drift detected!")
        # Log prod data as training data
        artifact = wandb.Artifact("training_data", type="dataset")
        artifact.add(wandb.Table(dataframe=prod_data), "training_data")
        artifact.description = prod_artifact.description
        run.log_artifact(artifact)
        # TODO: open a github issue asking for manual review
    else:
        print("No drift detected.")

    # Create a report explaining the drift (or lack thereof)
    report_url = make_report(
        run.entity, run.project, run.name, drift_detected, media_keys
    )

    # Link report to the run
    run.config["report_url"] = report_url

    # Add report to lineage
    with open("report.txt", "w") as f:
        f.write(report_url)

    art_type = "report"
    art_name = f"{run.name}-drift"
    report_artifact = wandb.Artifact(name=art_name, type=art_type)
    report_artifact.description = report_url
    report_artifact.add_file("report.txt")
    report_artifact = run.log_artifact(report_artifact)

    print(f"Drift report available at {report_url}")

    # Print the drift detection result in a parseable format
    print(f"DRIFT_DETECTED={drift_detected}")
