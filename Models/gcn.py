# Ref: https://github.com/tkipf/pygcn

import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, in_features, n_hidden, out_features, noise_dim, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(in_features, n_hidden)
        self.gc2 = GraphConvolution(n_hidden, n_hidden)
        self.gc3 = GraphConvolution(n_hidden, n_hidden - noise_dim)
        self.gc4 = GraphConvolution(n_hidden, n_hidden)
        self.gc5 = GraphConvolution(n_hidden, n_hidden)
        self.gc6 = GraphConvolution(n_hidden, n_hidden)
        self.gc7 = GraphConvolution(n_hidden, n_hidden)
        self.gc8 = GraphConvolution(n_hidden, n_hidden)
        self.gc9 = GraphConvolution(n_hidden, n_hidden)
        self.gc10 = GraphConvolution(n_hidden, out_features)
        self.dropout = dropout

    def forward(self, x, adj, noise):
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        x = F.relu(self.gc3(x, adj))
        noise = noise.unsqueeze(dim=0).repeat(x.shape[0], 1)
        x = torch.cat((x, noise), dim=1)
        x = F.relu(self.gc4(x, adj))
        x = F.relu(self.gc5(x, adj))
        x = F.relu(self.gc6(x, adj))
        x = F.relu(self.gc7(x, adj))
        x = F.relu(self.gc8(x, adj))
        x = F.relu(self.gc9(x, adj))
        x = self.gc10(x, adj)
        return torch.softmax(x, dim=1)
