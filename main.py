import argparse

import torch.cuda

from datasets import load_cora_data
from models import *
from utils import init_seed

args = argparse.ArgumentParser(description='arguments')
# data
args.add_argument('--dataset', default='Cora', type=str)

# model
args.add_argument('--model', default='GCN', type=str)
args.add_argument('--hidden-size', default=64, type=int)
args.add_argument('--cheb-order', default=3, type=int)

# train
args.add_argument('--lr', default=0.01, type=float)
args.add_argument('--epochs', default=50, type=int)
args.add_argument('--seed', default=50, type=int)
args.add_argument('--weight-decay', default=0.0, type=float)

args = args.parse_args()
init_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

load_data_func = None
in_size = 0
out_size = 0
if args.dataset == 'Cora':
    load_data_func = load_cora_data
    in_size = 1433
    out_size = 7
else:
    raise KeyError("Dataset {} is unknown.".format(args.dataset))

if args.model == 'GCN':
    model = GCN(in_size, args.hidden_size, out_size)
elif args.model == 'ChebNet':
    args.cheb_order = 4
    model = ChebNet(in_size, args.hidden_size, out_size, args.cheb_order)
elif args.model == 'GAT':
    args.hidden_size = 8
    args.weight_decay = 0.0005
    model = GAT(in_size, args.hidden_size, out_size)
elif args.model == 'JKNet':
    args.lr = 0.01
    args.weight_decay = 0.0005
    # model = JKNetMaxpool(in_size, out_size, n_layers=6, n_units=16,
    #                      aggregation='mean')
    model = JKNetConcat(in_size, out_size, n_layers=6, n_units=16,
                        aggregation='mean')
elif args.model == 'GraphSAGE':
    args.lr = 0.01
    args.hidden_size = 32
    args.weight_decay = 0.0005
    model = GraphSAGE(in_size, args.hidden_size, out_size, aggr='mean')
else:
    raise KeyError("Model {} is undefined.".format(args.model))

model = model.to(device)

train(model, load_data_func, init_lr=args.lr, epoch_num=args.epochs,
      weight_decay=args.weight_decay, device=device)
