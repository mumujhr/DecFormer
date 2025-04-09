import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import networkx as nx  # 用于计算最短路径


def calculate_calibration_mask(edge_index, num_nodes, lambda_val=0.5):
    # 创建一个空的图
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    # 添加边
    edges = edge_index.t().tolist()
    G.add_edges_from(edges)

    # 初始化最短路径距离矩阵
    shortest_path_matrix = torch.full((num_nodes, num_nodes), -1)

    # 计算最短路径距离
    for i in range(num_nodes):
        for j in range(num_nodes):
            if nx.has_path(G, i, j):
                shortest_path_matrix[i, j] = nx.shortest_path_length(G, i, j)

    # 从最短距离为3之后开始指数衰减
    # 最短距离小于3的设为1
    calibration_mask = torch.ones((num_nodes, num_nodes))

    # 处理最短距离大于等于3的情况
    mask = shortest_path_matrix >= 3
    # 最短距离超过10的统一按10处理
    adjusted_shortest_path = torch.clamp(shortest_path_matrix[mask], max=10)
    calibration_mask[mask] = lambda_val ** (adjusted_shortest_path - 2)

    return calibration_mask

