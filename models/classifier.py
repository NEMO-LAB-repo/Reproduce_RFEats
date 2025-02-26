import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 num_classes: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.encoder1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.ReLU(),
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.classifier(x)
        return x