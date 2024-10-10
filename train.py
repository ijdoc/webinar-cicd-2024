# train.property

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from simple_model import load_model
from utils import prep_time_series_data
import wandb

config = {
    "input_size": 4,
    "hidden_size": 28,
    "output_size": 1,
    "num_epochs": 10000,
    "batch_size": 64,
    "learning_rate": 0.000003,
    "validation_split": 0.2,
    "n_time_steps": 24,
}

wandb.init(
    project="wandb-webinar-cicd-2024",
    job_type="train",
    config=config,
)

artifact = wandb.use_artifact("jdoc-org/wandb-registry-dataset/training:latest")
df = artifact.get("training_data").get_dataframe()

# Prepare data (assumes the first column is the target value)
X = df.iloc[:, :].values  # All columns as input
y = df.iloc[:, 0].values.reshape(-1, 1)  # First column as target

# Normalize the data using StandardScaler
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Create time series data using n_time_steps
n_time_steps = config["n_time_steps"]
X_time_series, y_time_series = prep_time_series_data(X_scaled, y_scaled, n_time_steps)

# Convert time series data to tensors
X_tensor = torch.tensor(X_time_series, dtype=torch.float32)
y_tensor = torch.tensor(y_time_series, dtype=torch.float32)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_tensor, y_tensor, test_size=config["validation_split"], random_state=42
)

# Create DataLoaders for mini-batch training and validation
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

# Instantiate the model
model = load_model(
    input_size=config["input_size"] * config["n_time_steps"],
    hidden_size=config["hidden_size"],
    output_size=config["output_size"],
)

# Loss and optimizer
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

# Training loop with mini-batch training
best_val_loss = float("inf")  # Initialize a variable to track the best validation loss
for epoch in range(config["num_epochs"]):
    model.train()
    running_loss = 0.0

    # Loop over mini-batches
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # Validation loop
    model.eval()
    val_loss = 0.0
    val_mae_loss = 0.0
    val_r2_score = 0.0
    with torch.no_grad():
        ss_res = 0.0
        ss_tot = 0.0
        for batch_X, batch_y in val_loader:
            val_outputs = model(batch_X)
            loss = criterion(val_outputs, batch_y)
            val_loss += loss.item()

            # Calculate MAE
            val_mae_loss += F.l1_loss(val_outputs, batch_y).item()

            # Calculate R²
            ss_res += torch.sum((batch_y - val_outputs) ** 2).item()
            ss_tot += torch.sum((batch_y - torch.mean(batch_y)) ** 2).item()

        avg_val_loss = val_loss / len(
            val_loader
        )  # Average validation loss for the epoch
        avg_val_mae = val_mae_loss / len(val_loader)  # Average validation MAE
        val_r2_score = 1 - (ss_res / ss_tot)  # Validation R²

    wandb.log(
        {
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_mae": avg_val_mae,
            "val_r2": val_r2_score,
        }
    )
    # Print metrics every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(
            f'Epoch [{epoch+1}/{config["num_epochs"]}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val MAE: {avg_val_mae:.4f}, Val R²: {val_r2_score:.4f}'
        )

    # Save the best model based on validation loss
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        # Update best/summary metrics
        metrics = {
            "train_loss": avg_train_loss,
            "val_loss": avg_train_loss,
            "val_mae": avg_val_loss,
            "val_r2": val_r2_score,
        }
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "scaler_X": scaler_X,
                "scaler_y": scaler_y,
                "metrics": metrics,
                "config": config,
            },
            "best_model.pth",
        )

    wandb.summary.update(metrics)

    # print(f"Best model saved at epoch {epoch+1}")

# Save model as W&B artifact
artifact = wandb.Artifact("trained_model", type="model")
artifact.add_file("best_model.pth")
wandb.log_artifact(artifact)
wandb.finish()
