# Ref: https://github.com/tkipf/pygcn

import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.layers import GraphConvolution


class ResGCNFace(nn.Module):
    def __init__(self, in_features, n_hidden, out_features, noise_dim, dropout):
        super(ResGCNFace, self).__init__()
        self.gc1 = GraphConvolution(in_features, n_hidden)
        self.gc2 = GraphConvolution(n_hidden, n_hidden)
        self.gc3 = GraphConvolution(n_hidden, n_hidden)
        self.fc_noise = nn.Linear(n_hidden, n_hidden - noise_dim)
        self.fc1 = nn.Linear(n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_hidden)
        self.fc4 = nn.Linear(n_hidden, n_hidden)
        self.fc5 = nn.Linear(n_hidden, n_hidden)
        self.fc6 = nn.Linear(n_hidden, out_features)
        self.dropout = dropout

    def forward(self, x, adj, noise):
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        x = F.relu(self.gc3(x, adj))
        x = F.relu(self.fc_noise(x))
        noise = noise.unsqueeze(dim=0).repeat(x.shape[0], 1)
        x = torch.cat((x, noise), dim=1)
        x = F.relu(x + self.fc1(x))
        x = F.relu(x + self.fc2(x))
        x = F.relu(x + self.fc3(x))
        x = F.relu(x + self.fc4(x))
        x = F.relu(x + self.fc5(x))
        x = self.fc6(x)
        return torch.softmax(x, dim=1)
