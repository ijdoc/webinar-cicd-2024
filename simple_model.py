import torch
import torch.nn as nn


class SimpleDNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.2):
        super(SimpleDNN, self).__init__()
        self.half_hidden_size = round(hidden_size / 2)

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)

        self.fc2 = nn.Linear(hidden_size, self.half_hidden_size)
        self.bn2 = nn.BatchNorm1d(self.half_hidden_size)
        self.fc3 = nn.Linear(self.half_hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)

        self.fc4 = nn.Linear(hidden_size, output_size)  # Output layer

        # Activation function
        self.relu = nn.ReLU()

        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # First hidden layer
        out = self.fc1(x)
        out = self.bn1(out)  # Apply batch normalization
        out = self.relu(out)
        out = self.dropout(out)  # Apply dropout

        # Second hidden layer
        out = self.fc2(out)
        out = self.bn2(out)  # Apply batch normalization
        out = self.relu(out)

        # Third hidden layer
        out = self.fc3(out)
        out = self.bn3(out)  # Apply batch normalization
        out = self.relu(out)

        # Output layer (no activation here, typically applied outside for regression/classification)
        out = self.fc4(out)
        return out


def load_model(input_size, hidden_size, output_size, dropout_prob=0.2):
    model = SimpleDNN(input_size, hidden_size, output_size, dropout_prob)
    return model
