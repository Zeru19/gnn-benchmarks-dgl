import dgl.function as fn
from dgl.nn.pytorch.conv import ChebConv
import torch.nn as nn


# class ChebNet(nn.Module):
#     def __init__(self, in_size, out_size, cheb_order):
#         super().__init__()
#         self.cheb_conv = ChebConv(in_size, out_size, k=cheb_order)
#
#     def forward(self, g, features):
#         x = self.cheb_conv(g, features)
#         return x


class ChebNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, cheb_order):
        super().__init__()
        self.cheb_conv1 = ChebConv(in_size, hidden_size, k=cheb_order)
        self.cheb_conv2 = ChebConv(hidden_size, out_size, k=cheb_order)
        self.relu = nn.ReLU()

    def forward(self, g, features):
        x = self.relu(self.cheb_conv1(g, features))
        x = self.cheb_conv2(g, x)
        return x
