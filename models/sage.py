from dgl.nn.pytorch.conv import SAGEConv
import torch
from torch import nn


class GraphSAGE(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, aggr='mean'):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_size, hidden_size, aggr, activation=nn.ReLU()))
        self.layers.append(SAGEConv(hidden_size, out_size, aggr, activation=None))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, g, features):
        x = features
        for layer in self.layers:
            x = layer(g, x)
        return x
