from dgl.nn.pytorch.conv import GATConv
import torch.nn as nn


class GAT(nn.Module):
    """
    Standard GAT for Cora
    """

    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        num_heads = 8
        self.gat_conv1 = GATConv(in_size, hidden_size, num_heads=num_heads,
                                 feat_drop=0.6, attn_drop=0.6, residual=True,
                                 activation=nn.ELU())
        self.gat_conv2 = GATConv(hidden_size * num_heads, out_size, num_heads=1,
                                 feat_drop=0.6, attn_drop=0.6, residual=True,
                                 # activation=nn.Softmax
                                 )

    def forward(self, g, features):
        x = self.gat_conv1(g, features)  # (num_nodes, num_heads, hidden_size / num_heads)
        x = x.reshape(x.size(0), -1)
        x = self.gat_conv2(g, x)
        return x.reshape(x.size(0), -1)
