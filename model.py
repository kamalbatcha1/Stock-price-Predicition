import torch.nn as nn
import config


class LSTMModel(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT,
            batch_first=True
        )

        self.fc = nn.Linear(config.HIDDEN_SIZE, 1)

    def forward(self, x):

        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)