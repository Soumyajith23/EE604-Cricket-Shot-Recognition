
import torch
import torch.nn as nn
from torchvision import models


class ResNetLSTM(nn.Module):
    def __init__(
        self,
        num_classes=10,
        hidden_size=128,
        num_layers=1,
        dropout=0.3,
        bidirectional=False,
        pretrained=False,
    ):
        super().__init__()

        self.cnn = models.resnet50(weights=None)
        self.cnn.fc = nn.Identity()

        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(
            hidden_size * (2 if bidirectional else 1),
            num_classes,
        )

    def forward(self, x):
        batch, frames, channels, height, width = x.shape

        x = x.reshape(batch * frames, channels, height, width)
        x = self.cnn(x)
        x = x.reshape(batch, frames, -1)

        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])

        return self.fc(x)


# Support both model names
CricketShotModel = ResNetLSTM
