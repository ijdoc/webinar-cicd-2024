# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
import pandas as pd
from transformer_model import TimeSeriesTransformer
import wandb
import os

app = FastAPI()

# Load the model and normalization parameters
def load_model():
    # Authenticate with wandb
    wandb_api_key = os.getenv('WANDB_API_KEY')
    if not wandb_api_key:
        raise ValueError("WANDB_API_KEY environment variable not set")
    wandb.login(key=wandb_api_key)

    # Download the model artifact
    api = wandb.Api()
    artifact = api.artifact('YOUR_WANDB_ENTITY/YOUR_WANDB_PROJECT/trained_model:latest', type='model')
    artifact_dir = artifact.download(root='checkpoints')

    checkpoint_path = os.path.join(artifact_dir, 'model_checkpoint.pth')
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    # Load normalization parameters
    input_mean = pd.Series(checkpoint['input_mean'])
    input_std = pd.Series(checkpoint['input_std'])
    target_mean = checkpoint['target_mean']
    target_std = checkpoint['target_std']

    # Reconstruct the model
    model = TimeSeriesTransformer(
        src_input_dim=len(input_mean),
        tgt_input_dim=1,
        d_model=64,  # Should match your training config
        nhead=8,
        num_layers=3,
        dropout=0.1,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, input_mean, input_std, target_mean, target_std

model, input_mean, input_std, target_mean, target_std = load_model()

# Define request and response models
class PredictionRequest(BaseModel):
    src: list  # List of past time steps (history)
    tgt: list  # List of initial target inputs

class PredictionResponse(BaseModel):
    prediction: list  # Predicted future values

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    src = np.array(request.src)
    tgt = np.array(request.tgt)

    # Normalize input features
    src = (src - input_mean.values) / input_std.values

    # Normalize target variable
    tgt = (tgt - target_mean) / target_std

    src_tensor = torch.tensor(src).unsqueeze(0).float()  # Shape: (1, src_len, input_dim)
    tgt_tensor = torch.tensor(tgt).unsqueeze(0).float()  # Shape: (1, tgt_len, 1)

    with torch.no_grad():
        output = model(src_tensor, tgt_tensor)

    # Denormalize the output
    output = output.squeeze(0).numpy()
    output = output * target_std + target_mean

    return PredictionResponse(prediction=output.tolist())
