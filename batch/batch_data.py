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
            "batch_type": args.batch_type,  # The type of dataset to batch (training or production)
            "iteration": args.iteration,  # The iteration of the dataset to batch
            "history_days": args.history_days,  # The total length of the history batch in days
            "stride_days": args.stride_days,  # The number of days to stride between iterations
        },
    ) as run:

        batch_type = run.config.batch_type
        iteration = run.config.iteration
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

        # Convert the data column to a datetime index
        data["date"] = pd.to_datetime(data["date"])
        data = data.set_index("date")

        # Resample the data to daily frequency ('D'), and calculate the mean for each day
        daily_data = data.resample("D").mean()

        # Grab history_days of data, offset by iteration and stride_days (if production)
        if batch_type == "training":
            offset = stride_days * iteration
        elif batch_type == "production":
            offset = stride_days * (iteration + 1)
        data = daily_data.iloc[offset : offset + history_days]

        # Log a summary of the data
        run.log(
            {
                "data_summary": data.describe().reset_index(),
                "data_sample": data.head().reset_index(),
            }
        )

        # Log the dataset
        artifact = wandb.Artifact(f"{batch_type}_data", type="dataset")
        artifact.add(wandb.Table(dataframe=data), f"{batch_type}_data")
        artifact.description = (
            f"Batched {batch_type} data for iteration {iteration}. "
            f"This iteration simulated data collection over the previous "
            f"{history_days} days, with a stride of {stride_days} days"
        )
        artifact = run.log_artifact(artifact).wait()
        print(
            f"{batch_type} data logged as {artifact.source_name}."
            f"- Iteration: {iteration}\n"
            f"- Iteration Stride: {stride_days} day(s)\n"
            f"- Total length: {history_days} day(s)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-type", type=str, default="training", choices=["training", "production"]
    )
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--history-days", type=int, default=200)
    parser.add_argument("--stride-days", type=int, default=1)

    main(parser.parse_args())
