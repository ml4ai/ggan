# Ref: https://github.com/tkipf/pygcn

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeRF(nn.Module):
    def __init__(self, in_features, n_hidden, out_features, noise_dim, dropout):
        super(NeRF, self).__init__()
        self.fc_noise = nn.Linear(in_features + noise_dim, n_hidden)
        self.fc1 = nn.Linear(n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_hidden)
        self.fc4 = nn.Linear(n_hidden, n_hidden)
        self.fc5 = nn.Linear(n_hidden, n_hidden)
        self.fc6 = nn.Linear(n_hidden, out_features)
        self.dropout = dropout

    def forward(self, x, adj, noise):
        noise = noise.unsqueeze(dim=0).repeat(x.shape[0], 1)
        x = torch.cat((x, noise), dim=1)
        x = F.relu(self.fc_noise(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return torch.softmax(x, dim=1)
