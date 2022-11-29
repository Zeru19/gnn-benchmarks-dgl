import dgl.function as fn
import torch
import torch.nn.functional as F

AGGREGATIONS = {
    'sum': torch.sum,
    'mean': torch.mean,
    'max': torch.max,
}


class GraphConvLayer(torch.nn.Module):
    """Graph convolution layer.

    Args:
        in_features (int): Size of each input node.
        out_features (int): Size of each output node.
        aggregation (str): 'sum', 'mean' or 'max'.
                           Specify the way to aggregate the neighbourhoods.
    """
    def __init__(self, in_features, out_features, aggregation='sum'):
        super(GraphConvLayer, self).__init__()

        if aggregation not in AGGREGATIONS.keys():
            raise ValueError("'aggregation' argument has to be one of "
                             "'sum', 'mean' or 'max'.")
        self.aggregate = lambda nodes: AGGREGATIONS[aggregation](nodes, dim=1)

        self.linear = torch.nn.Linear(in_features, out_features)
        self.self_loop_w = torch.nn.Linear(in_features, out_features)
        self.bias = torch.nn.Parameter(torch.zeros(out_features))

    def forward(self, graph, x):
        graph.ndata['h'] = x
        graph.update_all(
            fn.copy_src(src='h', out='msg'),
            lambda nodes: {'h': self.aggregate(nodes.mailbox['msg'])})
        h = graph.ndata.pop('h')
        h = self.linear(h)
        return h + self.self_loop_w(x) + self.bias


class JKNetConcat(torch.nn.Module):
    """An implementation of Jumping Knowledge Network (arxiv 1806.03536) which
    combine layers with concatenation.

    Args:
        in_features (int): Size of each input node.
        out_features (int): Size of each output node.
        n_layers (int): Number of the convolution layers.
        n_units (int): Size of the middle layers.
        aggregation (str): 'sum', 'mean' or 'max'.
                           Specify the way to aggregate the neighbourhoods.
    """
    def __init__(self, in_features, out_features, n_layers=6, n_units=16,
                 aggregation='sum'):
        super(JKNetConcat, self).__init__()
        self.n_layers = n_layers

        self.gconv0 = GraphConvLayer(in_features, n_units, aggregation)
        self.dropout0 = torch.nn.Dropout(0.5)
        for i in range(1, self.n_layers):
            setattr(self, 'gconv{}'.format(i),
                    GraphConvLayer(n_units, n_units, aggregation))
            setattr(self, 'dropout{}'.format(i), torch.nn.Dropout(0.5))
        self.last_linear = torch.nn.Linear(n_layers * n_units, out_features)

    def forward(self, graph, x):
        layer_outputs = []
        for i in range(self.n_layers):
            dropout = getattr(self, 'dropout{}'.format(i))
            gconv = getattr(self, 'gconv{}'.format(i))
            x = dropout(F.relu(gconv(graph, x)))
            layer_outputs.append(x)

        h = torch.cat(layer_outputs, dim=1)
        return self.last_linear(h)


class JKNetMaxpool(torch.nn.Module):
    """An implementation of Jumping Knowledge Network (arxiv 1806.03536) which
    combine layers with Maxpool.

    Args:
        in_features (int): Size of each input node.
        out_features (int): Size of each output node.
        n_layers (int): Number of the convolution layers.
        n_units (int): Size of the middle layers.
        aggregation (str): 'sum', 'mean' or 'max'.
                           Specify the way to aggregate the neighbourhoods.
    """
    def __init__(self, in_features, out_features, n_layers=6, n_units=16,
                 aggregation='sum'):
        super(JKNetMaxpool, self).__init__()
        self.n_layers = n_layers

        self.gconv0 = GraphConvLayer(in_features, n_units, aggregation)
        self.dropout0 = torch.nn.Dropout(0.5)
        for i in range(1, self.n_layers):
            setattr(self, 'gconv{}'.format(i),
                    GraphConvLayer(n_units, n_units, aggregation))
            setattr(self, 'dropout{}'.format(i), torch.nn.Dropout(0.5))
        self.last_linear = torch.nn.Linear(n_units, out_features)

    def forward(self, graph, x):
        layer_outputs = []
        for i in range(self.n_layers):
            dropout = getattr(self, 'dropout{}'.format(i))
            gconv = getattr(self, 'gconv{}'.format(i))
            x = dropout(F.relu(gconv(graph, x)))
            layer_outputs.append(x)

        h = torch.stack(layer_outputs, dim=0)
        h = torch.max(h, dim=0)[0]
        return self.last_linear(h)
