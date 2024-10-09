import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import pandas as pd
from simple_model import load_model
import wandb
import os
from utils import plot_predictions_vs_actuals, prep_time_series_data
import numpy as np

# Initialize W&B for evaluation job
wandb.init(
    project="wandb-webinar-cicd-2024",
    job_type="evaluate",
)

# Load the latest model artifact
registered_production_model = "jdoc-org/wandb-registry-model/production:latest"
model_artifact = wandb.use_artifact(registered_production_model)
wandb.config["model"] = model_artifact.name
model_dir = model_artifact.download()
model_path = os.path.join(model_dir, "best_model.pth")

# Load the checkpoint
checkpoint = torch.load(model_path, map_location=torch.device("cpu"))

# Load the saved scalers and metrics
scaler_X = checkpoint["scaler_X"]
scaler_y = checkpoint["scaler_y"]
metrics = checkpoint["metrics"]
config = checkpoint["config"]

# Instantiate the model and load its state dictionary
model = load_model(
    config["input_size"] * config["n_time_steps"],
    config["hidden_size"],
    config["output_size"],
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()  # Set the model to evaluation mode

# Load the test data artifact from W&B
artifact = wandb.use_artifact("jdoc-org/wandb-registry-dataset/training:latest")
df_test = artifact.get("training_data").get_dataframe()

# Prepare data (assumes the first column is the target value)
X_test = df_test.iloc[:, :].values  # Last 3 columns as input
y_test = df_test.iloc[:, 0].values.reshape(-1, 1)  # First column as target

# Normalize the data using StandardScaler
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# Normalize the data using StandardScaler
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_test_scaled = scaler_X.fit_transform(X_test)
y_test_scaled = scaler_y.fit_transform(y_test)

# Create time series data using n_time_steps
n_time_steps = config["n_time_steps"]
X_time_series, y_time_series = prep_time_series_data(
    X_test_scaled, y_test_scaled, config["n_time_steps"]
)

# Convert time series data to tensors
X_test_tensor = torch.tensor(X_time_series, dtype=torch.float32)
y_test_tensor = torch.tensor(y_time_series, dtype=torch.float32)

# Create a DataLoader for the test data
test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config["batch_size"])

# Evaluation loop
mse_loss = 0.0
mae_loss = 0.0
ss_res = 0.0
ss_tot = 0.0
all_predictions = []
all_actuals = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)

        # Store predictions and actuals for plotting
        all_predictions.append(outputs.numpy())
        all_actuals.append(batch_y.numpy())

        # Calculate MSE
        mse_loss += F.mse_loss(outputs, batch_y).item()

        # Calculate MAE
        mae_loss += F.l1_loss(outputs, batch_y).item()

        # Calculate R²
        ss_res += torch.sum((batch_y - outputs) ** 2).item()
        ss_tot += torch.sum((batch_y - torch.mean(batch_y)) ** 2).item()

# Average the losses over the test dataset
mse_loss /= len(test_loader)
mae_loss /= len(test_loader)
r2_score = 1 - (ss_res / ss_tot)

# Log evaluation metrics to W&B
eval_table = wandb.Table(columns=["Metric", "Validation", "Evaluation"])
eval_table.add_data("MSE", metrics["val_loss"], mse_loss)
eval_table.add_data("MAE", metrics["val_mae"], mae_loss)
eval_table.add_data("R²", metrics["val_r2"], r2_score)
wandb.log({"performance_metrics": eval_table})

# Convert predictions and actuals to numpy arrays for plotting
all_predictions = scaler_y.inverse_transform(np.vstack(all_predictions))
all_actuals = scaler_y.inverse_transform(np.vstack(all_actuals))

# Generate and log predictions vs actuals plot
plt = plot_predictions_vs_actuals(all_actuals, all_predictions)
wandb.log({"predictions_vs_actuals": wandb.Image(plt)})

# Print metrics to console
print(f"Test MSE: {mse_loss:.4f}")
print(f"Test MAE: {mae_loss:.4f}")
print(f"Test R²: {r2_score:.4f}")

# Finish W&B run
wandb.finish()
