import torch
import torch.nn.functional as F
from sklearn.metrics import recall_score, f1_score
import time
from model import GCN
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix
import argparse
import dgl
import torch.nn.functional as F
import numpy as np
import warnings
def train(g, model, epochs,lr):
    optimizer = torch.optim.Adam(model.parameters(), lr)
    best_val_acc = 0
    best_test_acc = 0
    best_recall = 0
    best_f1 = 0
    best_auc = 0
    best_g_means = 0

    features = g.ndata['feat']
    labels = g.ndata['labels']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    start_time = time.time()

    for epoch in range(epochs):
        # Forward
        logits = model(g, features)

        # Compute prediction
        pred = logits.argmax(1)

        # Compute loss
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        # Compute accuracy on training/validation/test
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # Compute recall and F1 score
        train_recall = recall_score(labels[train_mask].cpu(), pred[train_mask].cpu())
        train_f1 = f1_score(labels[train_mask].cpu(), pred[train_mask].cpu())
        val_recall = recall_score(labels[val_mask].cpu(), pred[val_mask].cpu())
        val_f1 = f1_score(labels[val_mask].cpu(), pred[val_mask].cpu())
        test_recall = recall_score(labels[test_mask].cpu(), pred[test_mask].cpu())
        test_f1 = f1_score(labels[test_mask].cpu(), pred[test_mask].cpu())

        # Compute AUC
        y_true = labels[test_mask].cpu().numpy()
        y_scores = F.softmax(logits[test_mask], dim=1).cpu().detach().numpy()[:, 1]
        auc_score = roc_auc_score(y_true, y_scores)

        tn, fp, fn, tp = confusion_matrix(y_true, pred[test_mask], labels=[0, 1]).ravel()
        g_means = np.sqrt((tp / (tp + fn)) * (tn / (tn + fp)))

        # Save the best validation accuracy and the corresponding test accuracy, recall, F1 score, AUC, and G-means.
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        if test_recall > best_recall:
            best_recall = test_recall

        if test_f1 > best_f1:
            best_f1 = test_f1

        if auc_score > best_auc:
            best_auc = auc_score

        if g_means > best_g_means:
            best_g_means = g_means

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print("In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})".format(
                epoch, loss, val_acc, best_val_acc, test_acc, best_test_acc
            ))
            print("Recall: train {:.3f}, val {:.3f}, test {:.3f}".format(train_recall, val_recall, test_recall))
            print("F1 Score: train {:.3f}, val {:.3f}, test {:.3f}".format(train_f1, val_f1, test_f1))
            print("AUC: {:.3f}, G-means: {:.3f}".format(auc_score, g_means))

    end_time = time.time()
    total_time = end_time - start_time
    print("Total training time: {:.2f} seconds".format(total_time))
    
    print("Best results - Val Acc: {:.3f}, Test Acc: {:.3f}, Recall: {:.3f}, F1 Score: {:.3f}, AUC: {:.3f}, G-means: {:.3f}".format(
        best_val_acc, best_test_acc, best_recall, best_f1, best_auc, best_g_means
    ))


def get_args():
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--lr','-lr',type=int,default=0.001)
    parser.add_argument('--device','-device',type=str,default='cpu')
    parser.add_argument('--epochs','-epochs',type=int,default=1000)
    parser.add_argument('--graph','-graph',type=str,default='ccfraud_graph.bin')
    return parser.parse_args()

if __name__ == "__main__":
    
    warnings.filterwarnings("ignore", category=UserWarning)

    args = get_args()
    # Load the saved graph
    loaded_graphs, _ = dgl.load_graphs(args.graph)
    loaded_g = loaded_graphs[0]
    print("Graph loaded successfully.")
    args = get_args()
    device = args.device
    model = GCN(in_feats=16,num_classes=2)
    model.to(device)
    loaded_g.to(device)
    nepoch = args.epochs
    lr = args.lr
    train(loaded_g,model,nepoch,lr)