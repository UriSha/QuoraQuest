import torch.nn as nn
import torch.nn.functional as F


class MasterClassifier(nn.Module):
    def __init__(self, input_dim):
        super(MasterClassifier, self).__init__()

        # input_dim is the dimensions of two concatenated questions
        self.fc1 = nn.Linear(input_dim, 300)
        self.fc2 = nn.Linear(300, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.sigmoid(X)
