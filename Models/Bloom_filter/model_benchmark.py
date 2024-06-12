import torch.nn as nn

class model_benchmark(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_size = 256
        self.model = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        # self.l4 = torch.nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, train=True):
        opt = self.model(inputs)
        return self.sigmoid(opt)

