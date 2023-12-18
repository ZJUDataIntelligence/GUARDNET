import dgl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# 定义 MLP 模型
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size1=512, hidden_size2=256):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()  # 使用ReLU激活函数
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()  # 使用ReLU激活函数
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


class GCN(nn.Module):
    def __init__(self, in_feats, num_classes, hidden_size1=256, hidden_size2=128):
        super(GCN, self).__init__()
        self.conv1 = dgl.nn.GraphConv(in_feats, hidden_size1,allow_zero_in_degree=True)
        self.conv2 = dgl.nn.GraphConv(hidden_size1, hidden_size2,allow_zero_in_degree=True)
        self.conv3 = dgl.nn.GraphConv(hidden_size2, num_classes,allow_zero_in_degree=True)

    def forward(self, g, features):
        h = self.conv1(g, features)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        h = self.conv3(g, h)
        return h