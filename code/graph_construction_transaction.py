import dgl
import pandas as pd
import numpy as np
import torch
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import concurrent.futures
import argparse
from model import MLP
class Minor_node_centred_Subgraph:
    def __init__(self, graph, dataframe,k_rw,k_khop,k_ego):
        self.graph = graph
        self.dataframe = dataframe
        self.k_rw = k_rw
        self.k_khop = k_khop
        self.k_ego = k_ego
    def get_fraud_subgraph(self):
        fraud_nodes = np.where(np.array(self.dataframe['isFraud']) == 1)[0]

        # # Random walk nodes
        rw_nodes = dgl.sampling.random_walk(self.graph, fraud_nodes, length=self.k_rw)[0]
        rw_nodes = np.unique(rw_nodes.flatten())

        # # Node subgraph
        subgraph = dgl.node_subgraph(self.graph, rw_nodes[1:])

        # k-hop subgraph
        sg, inverse_indices = dgl.khop_in_subgraph(self.graph, fraud_nodes, k=self.k_khop)

        # All neighbors
        all_neighbors = [self.graph.successors(node) for node in fraud_nodes]
        merged_tensor = np.concatenate(all_neighbors)
        all_neighbors = np.unique(merged_tensor.flatten())

        # k-ego subgraph
        ego_subgraph, inverse_indices = dgl.khop_in_subgraph(self.graph, all_neighbors, k=self.k_ego)

        # Batch graphs
        graphs = [subgraph, ego_subgraph,sg]
        merged_graph = dgl.batch(graphs)
        if '_ID' in merged_graph.edata:
            del merged_graph.edata['_ID']
        if '_ID' in merged_graph.ndata:
            del merged_graph.ndata['_ID']
        return merged_graph




class Major_node_centred_downsampling:
    def __init__(self, graph, gamma,input_size):
        self.graph = graph
        self.gamma = gamma
        self.input_size = input_size

    def apply_mlp(self):
        class_0_nodes = (self.graph.ndata['labels'] == 0).nonzero().squeeze()

        if len(class_0_nodes) == 0:
            return self.graph

        class_0_features = self.graph.ndata['feat'][class_0_nodes]

        mlp_model = MLP(input_size=self.input_size, output_size=2)  # 2 表示输出两类，keep 和 drop

        predictions = mlp_model(class_0_features)

        sampled_nodes = torch.bernoulli(torch.full_like(predictions[:, 1], 1 - self.gamma)).nonzero().squeeze()

        if len(sampled_nodes) == 0:
            return dgl.graph()

        self.graph.remove_nodes(class_0_nodes[sampled_nodes])

        return self.graph
def get_args():
    parser = argparse.ArgumentParser(description='Graph construction')
    parser.add_argument('--k_rw','-k_rw',type=int,default=10)
    parser.add_argument('--k_khop','-k_khop',type=int,default=2)
    parser.add_argument('--k_ego','-k_ego',type=int,default=3)
    parser.add_argument('--gamma','-gamma',type=int,default=0.6)
    parser.add_argument('--input_size','-input_size',type=int,default=28)

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    # 读取数据集
    df = pd.read_csv(r"fakejob.csv")
    print("Data loaded.")
    features = np.array(df.drop(columns=['is_fraud']))


   # 使用KNeighborsClassifier计算KNN相似度
    knn = NearestNeighbors(n_neighbors=10, metric='euclidean')
    knn.fit(features)
    # 计算KNN图的相似度矩阵
    similarity_sparse = knn.kneighbors_graph(features, mode='distance')  # 返回稀疏矩阵

    # 将稀疏矩阵转换为稀疏张量
    rows, cols = similarity_sparse.nonzero()
    values = similarity_sparse.data

    # 构建边列表，相似度小于阈值的节点之间连接一条边
    threshold = 0.8  # 阈值
    edges = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        def process_edge(i):
            if values[i] < threshold:
                edges.append((rows[i], cols[i]))

        futures = [executor.submit(process_edge, i) for i in range(len(rows))]
        concurrent.futures.wait(futures)

    # 转换为numpy数组并进行转置
    edges = np.array(edges).T

    # 构建图
    g = dgl.graph((edges[0], edges[1]), num_nodes=len(total))
    # 随机生成train_mask、val_mask、test_mask
    n = len(df)
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    train_mask[:n // 2] = True
    val_mask[n // 2:3 * n // 4] = True
    test_mask[3 * n // 4:] = True
    g.ndata['train_mask'] = torch.tensor(train_mask)
    g.ndata['val_mask'] = torch.tensor(val_mask)
    g.ndata['test_mask'] = torch.tensor(test_mask)

    # 设置每个节点的特征
    g.ndata['feat'] = torch.FloatTensor(features)
    g.ndata['labels'] = torch.LongTensor(np.array(df['is_fraud']))


    # 使用例子
    # graph 和 df 是你的图和数据框
    processor = Minor_node_centred_Subgraph(g, df,args.k_rw,args.k_khop,args.k_ego)
    result_graph = processor.get_fraud_subgraph()
    sampler = Major_node_centred_downsampling(g, args.gamma,args.input_size)
    sampled_graph = sampler.apply_mlp()
    graphs = [sampled_graph, result_graph]
    g = dgl.batch(graphs)
    dgl.save_graphs("ieeecis_graph.bin", [g])
    print(g)
    print("Graph saved successfully.")



