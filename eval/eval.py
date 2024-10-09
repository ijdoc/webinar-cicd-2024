# check_model_degradation.py

import wandb
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from transformer_model import TimeSeriesTransformer  # Ensure this is accessible
from utils import plot_predictions_vs_actuals
import os

# Initialize a W&B run
with wandb.init(
    project="wandb-webinar-cicd-2024",
    job_type="evaluate",
) as run:

    # Load the latest model artifact
    registered_production_model = "jdoc-org/wandb-registry-model/production:latest"
    model_artifact = run.use_artifact(registered_production_model)
    run.config["model"] = model_artifact.name
    model_dir = model_artifact.download()
    model_path = os.path.join(model_dir, "model_checkpoint.pth")

    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))

    # Extract the normalization parameters
    input_mean = pd.Series(checkpoint["input_mean"])
    input_std = pd.Series(checkpoint["input_std"])
    target_mean = checkpoint["target_mean"]
    target_std = checkpoint["target_std"]
    model_dim = checkpoint["model_dim"]
    num_heads = checkpoint["num_heads"]
    num_layers = checkpoint["num_layers"]
    dropout_prob = checkpoint["dropout_prob"]
    src_len = checkpoint["src_len"]
    tgt_len = checkpoint["tgt_len"]
    batch_size = checkpoint["batch_size"]

    input_columns = ["temp", "humidity", "pressure", "active_power"]
    target_column = "active_power"
    input_dim = len(input_columns)

    # Initialize the model
    model = TimeSeriesTransformer(
        src_input_dim=input_dim,
        tgt_input_dim=1,
        d_model=model_dim,
        nhead=num_heads,
        num_layers=num_layers,
        dropout=dropout_prob,
    )

    # Load the model state
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # Set the model to evaluation mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 2. Load the latest production data
    # prod_artifact = run.use_artifact("production_data:latest")
    # run.config["prod_data"] = prod_artifact.name
    # data = prod_artifact.get("production_data").get_dataframe()
    # 2. Load the original training data
    prod_artifact = run.use_artifact("training_data:v3")
    run.config["prod_data"] = prod_artifact.name
    data = prod_artifact.get("training_data").get_dataframe()

    # 3. Preprocess the data using the same normalization parameters
    # Normalize input features
    data[input_columns] = (data[input_columns] - input_mean) / input_std

    # Normalize target variable
    data[target_column] = (data[target_column] - target_mean) / target_std

    num_days = data.shape[0]  # Total number of days
    num_samples = num_days - src_len - tgt_len + 1

    if num_samples <= 0:
        raise ValueError("Not enough data to create sequences for the model.")

    # Prepare source and target data
    input_data = data[input_columns].values  # (num_days, input_dim)
    target_data = data[target_column].values  # (num_days,)

    # Prepare source and target tensors
    src_data_np = np.array([input_data[i : i + src_len] for i in range(num_samples)])
    src_data = torch.from_numpy(
        src_data_np
    ).float()  # (num_samples, src_len, input_dim)

    tgt_data_np = np.array(
        [target_data[i + src_len : i + src_len + tgt_len] for i in range(num_samples)]
    )
    tgt_data = (
        torch.from_numpy(tgt_data_np).unsqueeze(-1).float()
    )  # (num_samples, tgt_len, 1)

    # Move data to device
    src_data = src_data.to(device)
    tgt_data = tgt_data.to(device)

    # 4. Make predictions
    model.eval()
    with torch.no_grad():
        predictions = []
        actuals = []
        for i in range(0, num_samples, batch_size):
            src_batch = src_data[i : i + batch_size]  # (batch_size, src_len, input_dim)
            tgt_batch = tgt_data[i : i + batch_size]  # (batch_size, tgt_len, 1)

            # Prepare the target input (shifted by one time step)
            tgt_input = tgt_batch[:, :-1, :]  # Remove the last time step

            # Forward pass
            output = model(src_batch, tgt_input)  # (batch_size, tgt_len - 1, 1)

            # Collect predictions and actuals
            predictions.append(output.cpu().numpy())
            actuals.append(tgt_batch[:, 1:, :].cpu().numpy())

    # Concatenate predictions and actuals
    predictions = np.concatenate(predictions, axis=0)  # (num_samples, tgt_len - 1, 1)
    actuals = np.concatenate(actuals, axis=0)  # (num_samples, tgt_len - 1, 1)

    # Reshape for metric calculations
    predictions = predictions.reshape(-1)
    actuals = actuals.reshape(-1)

    # Denormalize the predictions and actuals
    predictions = predictions * target_std + target_mean
    actuals = actuals * target_std + target_mean

    # 5. Calculate performance metrics
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)

    # Log the metrics to W&B
    metrics = {
        "mean_squared_error": mse,
        "mean_absolute_error": mae,
        "r2_score": r2,
    }
    run.log(metrics)

    # 6. Compare with baseline or threshold
    # Define your baseline metrics (from training or validation)
    # Replace these values with your actual baseline metrics
    baseline_metrics = {
        "mean_squared_error": 0.02,  # Baseline MSE
        "mean_absolute_error": 0.01,  # Baseline MAE
        "r2_score": 0.95,  # Baseline R² score
    }

    # Define acceptable degradation thresholds
    degradation_thresholds = {
        "mean_squared_error": 0.01,  # Acceptable increase in MSE
        "mean_absolute_error": 0.005,  # Acceptable increase in MAE
        "r2_score": -0.02,  # Acceptable decrease in R² score
    }

    # Check for degradation
    degradation_detected = False
    degradation_details = {}

    for metric_name in metrics:
        metric_value = metrics[metric_name]
        baseline_value = baseline_metrics[metric_name]
        threshold = degradation_thresholds[metric_name]

        if metric_name in ["mean_squared_error", "mean_absolute_error"]:
            # For error metrics, higher value indicates worse performance
            degradation = metric_value - baseline_value
            if degradation > threshold:
                degradation_detected = True
                degradation_details[metric_name] = degradation
        elif metric_name == "r2_score":
            # For R² score, lower value indicates worse performance
            degradation = baseline_value - metric_value
            if degradation > abs(threshold):
                degradation_detected = True
                degradation_details[metric_name] = degradation

    # 7. Log predictions vs actuals plot
    plt = plot_predictions_vs_actuals(actuals, predictions)
    run.log({"predictions_vs_actuals": wandb.Image(plt)})
    plt.close()

    # 8. Handle degradation detection
    if degradation_detected:
        print("> [!WARNING]")
        print("> Model degradation detected.\n")

    else:
        print("> No significant model degradation detected.\n")

    # 9. Log the run URL
    print(f"- [W&B Run]({run.url})")

    # Optionally, print the degradation detection result in a parseable format.
    # Useful for CI/CD pipelines.
    # print(f"DEGRADATION_DETECTED={degradation_detected}")
