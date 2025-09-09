import torch
import torch.nn.functional as F
from sklearn.metrics import recall_score, f1_score
import time
from model import GCN
from sklearn.metrics import roc_auc_score, confusion_matrix
import argparse
import dgl
import numpy as np
import warnings
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist

def get_args():
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--lr', '-lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', '-device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--epochs', '-epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--graph', '-graph', type=str, default='ccfraud_graph.bin', help='Graph file path')
    parser.add_argument('--save_path', '-save_path', type=str, default='./models/best_model.pth', help='Path to save the best model')
    return parser.parse_args()

def visualize_node_embeddings(model, g, features):
    plt.rcParams['font.family'] = 'Times New Roman'
    embeddings = model.get_node_embeddings(g, features).cpu().numpy()
    labels = g.ndata['labels']
    tsne = TSNE(n_components=2, perplexity=30, init='pca', n_iter=500)
    embeddings_2d = tsne.fit_transform(embeddings)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis')
    legend_labels = ['non-fraud', 'fraud']
    handles, _ = scatter.legend_elements()
    plt.legend(handles=handles, labels=legend_labels, title='Classes', fontsize=14, title_fontsize=16)
    plt.axis('off')
    plt.figtext(0.5, 0.01, 'd) GuardNet', ha='center', fontsize=22)
    plt.savefig('guardnet.pdf', format='pdf')
    plt.show()

def train(g, model, epochs, lr, save_path="best_model.pth"):
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    best_val_acc = 0
    best_test_acc = 0
    best_test_recall = 0
    best_test_f1 = 0
    best_test_g_means = 0
    best_model_state = None  # 用于保存最佳模型状态

    features = g.ndata['feat']
    labels = g.ndata['labels']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']

    # 类别权重
    class_counts = torch.bincount(labels[train_mask])
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum()

    # 初始化变量用于记录总时间
    total_time = 0
    epoch_times = []

    for epoch in range(epochs):
        # 记录 epoch 开始时间
        epoch_start_time = time.time()

        # Forward
        logits = model(g, features)
        pred = logits.argmax(1)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask], weight=class_weights.to(labels.device))

        # Compute accuracy
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # Compute recall and F1 score for test set
        test_recall = recall_score(labels[test_mask].cpu(), pred[test_mask].cpu(), zero_division=0)
        test_f1 = f1_score(labels[test_mask].cpu(), pred[test_mask].cpu(), zero_division=0)

        # Compute G-means for test set
        y_true = labels[test_mask].cpu().numpy()
        tn, fp, fn, tp = confusion_matrix(y_true, pred[test_mask].cpu().numpy(), labels=[0, 1]).ravel()
        test_g_means = np.sqrt((tp / (tp + fn)) * (tn / (tn + fp)))

        # Update best metrics
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_test_recall = test_recall
            best_test_f1 = test_f1
            best_test_g_means = test_g_means
            best_model_state = model.state_dict()  # 保存最佳模型

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # 记录 epoch 结束时间并计算耗时
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)
        total_time += epoch_time

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")
            print(f"Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}, Test G-means: {test_g_means:.4f}")
            print(f"Epoch Time: {epoch_time:.4f} seconds")

    # 计算平均 epoch 时间
    avg_epoch_time = total_time / epochs
    print(f"\nAverage epoch time: {avg_epoch_time:.4f} seconds")
    print(f"Total training time: {total_time:.4f} seconds")

    # 保存最佳模型
    if best_model_state:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(best_model_state, save_path)
        print(f"Best model saved to {save_path}")

    # 输出最佳测试结果
    print("\nBest Results:")
    print(f"Val Acc: {best_val_acc:.4f}, Test Acc: {best_test_acc:.4f}")
    print(f"Test Recall: {best_test_recall:.4f}, Test F1: {best_test_f1:.4f}, Test G-means: {best_test_g_means:.4f}")

    # 返回 epoch 时间列表，可用于进一步分析
    return epoch_times





if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    args = get_args()
    loaded_graphs, _ = dgl.load_graphs(args.graph)
    loaded_g = loaded_graphs[0]
    features = loaded_g.ndata['feat']
    current_feats = features.shape[1]  # 当前特征维度
    target_feats = current_feats  # 使用当前特征维度作为目标维度
    print("Graph loaded successfully.")
    device = args.device
    model = GCN(in_feats=target_feats, num_classes=2)
    model.to(device)
    loaded_g.to(device)

    # 接收 epoch 时间列表
    epoch_times = train(loaded_g, model, args.epochs, args.lr, save_path=args.save_path)

    # 可选：绘制 epoch 时间变化图
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_times)
    plt.title('Epoch Time Over Training')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    plt.savefig('epoch_times.png')
    plt.show()

