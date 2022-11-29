import dgl.function as fn
import torch.nn as nn


# the aggregation on a node u only involves
# summing over the neighbors’ representations hv
gcn_msg = fn.copy_u(u='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata['h']
            return self.linear(h)


class GCN(nn.Module):
    """
    Standard GCN
    """
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.layer1 = GCNLayer(in_size, hidden_size)
        self.layer2 = GCNLayer(hidden_size, out_size)
        self.relu = nn.ReLU()

    def forward(self, g, x):
        x = self.relu(self.layer1(g, x))
        x = self.layer2(g, x)
        return x


# class GCN(nn.Module):
#     """
#     Self-defined multi-layer GCN
#     """
#     def __init__(self, in_size, hidden_size, out_size):
#         super().__init__()
#         self.layers = nn.ModuleList()
#
#         if isinstance(hidden_size, int):
#             self.layers.append(GCNLayer(in_size, hidden_size))
#             self.layers.append(GCNLayer(hidden_size, out_size))
#         elif isinstance(hidden_size, tuple) or isinstance(hidden_size, list):
#             self.layers.append(GCNLayer(in_size, hidden_size[0]))
#             for i in range(len(hidden_size) - 1):
#                 self.layers.append(GCNLayer(hidden_size[i], hidden_size[i + 1]))
#             self.layers.append(GCNLayer(hidden_size[-1], out_size))
#         else:
#             raise TypeError("hidden_size type must be int, list or tuple")
#
#     def forward(self, g, x):
#         for layer in self.layers:
#             x = F.relu(layer(g, x))
#         return x
