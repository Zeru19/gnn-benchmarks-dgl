import time

import torch
import torch.nn.functional as F


def evaluate(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train(model, load_data_func, init_lr, epoch_num, weight_decay, device):
    g, features, labels, train_mask, test_mask = load_data_func()
    # add self link
    g.add_edges(g.nodes(), g.nodes())
    g, features, labels, train_mask, test_mask = \
        (e.to(device) for e in (g, features, labels, train_mask, test_mask))
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay)
    # criterion = torch.nn.NLLLoss()
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epoch_num):
        t0 = time.time()
        model.train()
        logits = model(g, features)
        logp = F.log_softmax(logits, 1)
        loss = criterion(logp[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = evaluate(model, g, features, labels, test_mask)
        print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
            epoch, loss.item(), acc, time.time() - t0))
