from dgl.data import CoraGraphDataset


def load_cora_data():
    dataset = CoraGraphDataset(raw_dir='datasets/')
    g = dataset[0]
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    test_mask = g.ndata['test_mask']
    return g, features, labels, train_mask, test_mask


if __name__ == '__main__':
    g, features, labels, train_mask, test_mask = load_cora_data()
    print(features, labels)
