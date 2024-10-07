import pandas as pd
import wandb
import seaborn as sns
import matplotlib.pyplot as plt


# Initialize W&B project
with wandb.init(
    # mode="disabled",
    project="wandb-webinar-cicd-2024",
    job_type="upload-raw",
    config={
        "paper_url": "https://doi.org/10.1016/j.dib.2024.110452",
        "dataset_url": "https://data.mendeley.com/datasets/tvhygj8rgg/1",
    },
) as run:

    # Load the dataset
    file_path = "energy_weather_raw_data.csv"
    data = pd.read_csv(file_path)

    # Grab the date values
    data["date"] = pd.to_datetime(data["date"])

    # Create & log temperature plot
    plt.figure(figsize=(10, 6))
    plt.plot(data["date"], data["temp"])
    plt.title("Temperature Over Time")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(True)
    # Save the box plot as an image file to speed up the workspace
    plot_path = "box_plot.png"
    plt.savefig(plot_path)
    # Log the image to W&B
    wandb.log({"Temperature Over Time": wandb.Image(plot_path)})

    # Create & log active power plot
    plt.figure(figsize=(10, 6))
    plt.plot(data["date"], data["active_power"])
    plt.title("Active Power Over Time")
    plt.xlabel("Date")
    plt.ylabel("Active Power (W)")
    plt.legend()
    plt.grid(True)
    # Save the box plot as an image file to speed up the workspace
    plot_path = "box_plot.png"
    plt.savefig(plot_path)
    # Log the image to W&B
    wandb.log({"Active Power Over Time": wandb.Image(plot_path)})

    # Box plot for the summary data (active_power, voltage, temp, humidity)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data[["active_power", "voltage"]])
    plt.title("Box Plot of Active Power and Voltage")
    # Save the box plot as an image file
    plot_path = "box_plot.png"
    plt.savefig(plot_path)
    # Log the image to W&B
    wandb.log({"Energy Box Plot": wandb.Image(plot_path)})

    # Box plot for the summary data (active_power, voltage, temp, humidity)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data[["temp", "humidity"]])
    plt.title("Box Plot of Temperature, and Humidity")
    # Save the box plot as an image file
    plot_path = "box_plot.png"
    plt.savefig(plot_path)
    # Log the image to W&B
    wandb.log({"Weather Box Plot": wandb.Image(plot_path)})

    # Calculate summary statistics for relevant columns
    summary_stats = data[["active_power", "voltage", "temp", "humidity"]].describe()
    # Reset the index to make it easier to log to W&B
    summary_stats = summary_stats.reset_index()
    print(summary_stats)

    # Log the dataset and paper URL as W&B artifacts
    artifact = wandb.Artifact("raw-data", type="dataset")
    artifact.add_file(file_path)
    artifact.add_file("data_in_brief.pdf")
    artifact.metadata["paper_url"] = run.config["paper_url"]  # Reference to the paper
    artifact.metadata["dataset_url"] = run.config[
        "dataset_url"
    ]  # Reference to the dataset
    wandb.log_artifact(artifact)

    # Log the plot to W&B
    wandb.log(
        {
            "data_sample": wandb.Table(dataframe=data.head()),
            "summary_statistics": summary_stats,
        }
    )
