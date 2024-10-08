# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from transformer_model import TimeSeriesTransformer
import wandb
import os
import numpy as np

with wandb.init(
    project="wandb-webinar-cicd-2024",
    job_type="train",
    config={
        "model_dim": 64,  # Transformer model dimension
        "num_heads": 8,  # Number of attention heads
        "num_layers": 3,  # Number of encoder layers
        "dropout_prob": 0.1,  # Dropout probability
        "learning_rate": 0.00005,  # Learning rate
        "epochs": 100,  # Number of training epochs
        "src_len": 30,  # Number of past time steps to use (history)
        "tgt_len": 7,  # Number of future time steps to predict
        "batch_size": 32,  # Batch size
    },
) as run:

    # Hyperparameters
    model_dim = run.config.model_dim
    num_heads = run.config.num_heads
    num_layers = run.config.num_layers
    dropout_prob = run.config.dropout_prob
    learning_rate = run.config.learning_rate
    epochs = run.config.epochs
    src_len = run.config.src_len
    tgt_len = run.config.tgt_len
    batch_size = run.config.batch_size

    # Grab the training dataset from the registry
    artifact = run.use_artifact("jdoc-org/wandb-registry-dataset/training:latest'")
    run.config["train_data"] = artifact.source_name
    data = artifact.get("training_data").get_dataframe()
    input_columns = ["temp", "humidity", "pressure", "active_power"]
    target_column = "active_power"
    input_dim = len(input_columns)
    num_days = data.shape[0]  # Total number of days
    print(f"Number of days: {num_days}")

    # **Normalization Step**

    # Compute mean and std for input features and target variable
    input_mean = data[input_columns].mean()
    input_std = data[input_columns].std()

    target_mean = data[target_column].mean()
    target_std = data[target_column].std()

    # Normalize input features
    data[input_columns] = (data[input_columns] - input_mean) / input_std

    # Normalize target variable
    data[target_column] = (data[target_column] - target_mean) / target_std

    # Instantiate the model with separate input dimensions
    model = TimeSeriesTransformer(
        src_input_dim=input_dim,
        tgt_input_dim=1,
        d_model=model_dim,
        nhead=num_heads,
        num_layers=num_layers,
        dropout=dropout_prob,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Training on {device}")

    # Loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error loss for regression
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Prepare source and target data for training
    num_samples = num_days - src_len - tgt_len + 1

    # Assume `data` is your dataset with multiple features
    input_data = data[input_columns].values  # (num_days, input_dim)
    target_data = data[target_column].values  # (num_days,)

    # Prepare source and target tensors without explicit loops
    # For src_data
    src_data_np = np.array([input_data[i : i + src_len] for i in range(num_samples)])
    src_data = torch.from_numpy(
        src_data_np
    ).float()  # Shape: (num_samples, src_len, input_dim)

    # For tgt_data
    tgt_data_np = np.array(
        [target_data[i + src_len : i + src_len + tgt_len] for i in range(num_samples)]
    )
    tgt_data = (
        torch.from_numpy(tgt_data_np).unsqueeze(-1).float()
    )  # Shape: (num_samples, tgt_len, 1)

    # Training loop
    for epoch in range(epochs):
        model.train()  # Set the model to training mode

        for i in range(0, num_samples, batch_size):
            # Get batch data
            src_batch = src_data[i : i + batch_size]  # (batch_size, src_len, input_dim)
            tgt_batch = tgt_data[i : i + batch_size]  # (batch_size, tgt_len, 1)
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            # Clear the gradients
            optimizer.zero_grad()

            # Prepare the target input (shifted by one time step)
            tgt_input = tgt_batch[
                :, :-1, :
            ]  # Remove the last time step from the target

            # Forward pass: predict future power consumption
            output = model(
                src_batch, tgt_input
            )  # Pass the source and the shifted target

            # Compute loss between predicted and actual values
            loss = criterion(
                output, tgt_batch[:, 1:, :]
            )  # Compare with the actual target

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Print the loss for this epoch
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        wandb.log({"loss": loss.item()})

    # Create a directory to save the model
    model_dir = "checkpoints"
    os.makedirs(model_dir, exist_ok=True)

    # Save model state and normalization parameters
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "input_mean": input_mean.to_dict(),
        "input_std": input_std.to_dict(),
        "target_mean": target_mean.item(),
        "target_std": target_std.item(),
    }
    checkpoint_path = os.path.join(model_dir, "model_checkpoint.pth")
    torch.save(checkpoint, checkpoint_path)

    # Create a wandb Artifact
    artifact = wandb.Artifact("trained_model", type="model")
    artifact.add_file(checkpoint_path)
    run.log_artifact(artifact)
