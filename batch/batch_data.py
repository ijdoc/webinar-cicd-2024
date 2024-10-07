import wandb
import os
import pandas as pd
import argparse


def main(args):

    with wandb.init(
        # mode="disabled",
        project="wandb-webinar-cicd-2024",
        job_type="batch-data",
        config={
            "type": args.type,  # The type of dataset to batch (training or production)
            "period_iteration": args.period_iteration,  # The iteration of the dataset to batch
            "history_days": args.history_days,  # The total length of the history batch in days
            "stride_days": args.stride_days,  # The number of days to stride between iterations
        },
    ) as run:

        type = run.config.type
        period_iteration = run.config.period_iteration
        history_days = run.config.history_days
        stride_days = run.config.stride_days

        # Grab complete historical dataset
        artifact = run.use_artifact("raw-data:latest")
        path_to_artifact = artifact.download()
        data = pd.read_csv(
            os.path.join(path_to_artifact, "energy_weather_raw_data.csv")
        )

        # Grab only the columns we want
        data = data[["date", "active_power", "temp", "humidity", "pressure"]]
        print(data.head())

        # Convert the data column to a datetime index
        data["date"] = pd.to_datetime(data["date"])
        data = data.set_index("date")

        # Resample the data to daily frequency ('D'), and calculate the mean for each day
        daily_data = data.resample("D").mean()

        # Grab history_days of data, offset by period_iteration and stride_days (if production)
        if type == "training":
            offset = period_iteration * history_days
        elif type == "production":
            offset = (period_iteration * history_days) + stride_days
        data = daily_data.iloc[offset : offset + history_days]

        print(data.head())

        # Log a summary of the data
        run.log(
            {
                "data_summary": data.describe().reset_index(),
                "data_sample": data.head().reset_index(),
            }
        )

        # Log the dataset
        artifact = wandb.Artifact(f"{type}_data", type="dataset")
        artifact.add(wandb.Table(dataframe=data), f"{type}_data")
        artifact.description = (
            f"Sampled {type} data for iteration {period_iteration}. "
            f"Iterations simulate data collection over the previous {history_days} "
            f"days, with a stride of {stride_days} days"
        )
        run.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type", type=str, default="training", choices=["training", "production"]
    )
    parser.add_argument("--period-iteration", type=int, default=0)
    parser.add_argument("--history-days", type=int, default=200)
    parser.add_argument("--stride-days", type=int, default=1)

    main(parser.parse_args())
