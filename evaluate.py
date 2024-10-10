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

# Load the production model
prod_model_name = "jdoc-org/wandb-registry-model/production:latest"
prod_artifact = wandb.use_artifact(prod_model_name)
wandb.config["prod_model"] = prod_artifact.name
prod_model_path = os.path.join(prod_artifact.download(), "best_model.pth")

# Load the rival model
rival_artifact = wandb.use_artifact("trained_model:latest")
wandb.config["rival_model"] = rival_artifact.name
rival_model_path = os.path.join(rival_artifact.download(), "best_model.pth")

# Load the checkpoint
model_checkpoint = torch.load(prod_model_path, map_location=torch.device("cpu"))
rival_checkpoint = torch.load(rival_model_path, map_location=torch.device("cpu"))

# Load the metrics for comparisson
prod_metrics = model_checkpoint["metrics"]

# Load rival scalers and metrics
scaler_X = rival_checkpoint["scaler_X"]
scaler_y = rival_checkpoint["scaler_y"]
config = rival_checkpoint["config"]
metrics = rival_checkpoint["metrics"]

# Instantiate the model and load its state dictionary
model = load_model(
    config["input_size"] * config["n_time_steps"],
    config["hidden_size"],
    config["output_size"],
)
model.load_state_dict(rival_checkpoint["model_state_dict"])
model.eval()  # Set the model to evaluation mode

# Load the latest production data artifact from W&B
artifact = wandb.use_artifact("production_data:latest")
df_test = artifact.get("production_data").get_dataframe()

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
eval_table = wandb.Table(columns=["Metric", "Production", "Candidate"])
eval_table.add_data("MSE", prod_metrics["val_loss"], mse_loss)
eval_table.add_data("MAE", prod_metrics["val_mae"], mae_loss)
eval_table.add_data("R²", prod_metrics["val_r2"], r2_score)
wandb.log({"performance_metrics": eval_table})

# Convert predictions and actuals to numpy arrays for plotting
all_predictions = scaler_y.inverse_transform(np.vstack(all_predictions))
all_actuals = scaler_y.inverse_transform(np.vstack(all_actuals))

# Generate and log predictions vs actuals plot
plt = plot_predictions_vs_actuals(all_actuals, all_predictions)
wandb.log({"predictions_vs_actuals": wandb.Image(plt)})

if prod_metrics["val_r2"] > r2_score:
    print("> Candidate model did not perform as well as the production model\n\n")
else:
    print("> [!INFO]")
    print("> The candidate model performed better than the production model\n\n")

    # Link the rival model to the proction model registry
    rival_artifact.link("jdoc-org/wandb-registry-model/production")
    print(
        "The candidate model has been promoted to the [production model registry](https://wandb.ai/registry/model?selectionPath=jdoc-org%2Fwandb-registry-model%2Fproduction&view=versions)!"
    )

print(f"- [W&B Run]({wandb.run.url})")

# Finish W&B run
wandb.finish()
