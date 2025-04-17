# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from tqdm import tqdm
# import networkx as nx
# import numpy as np
#
# #
# # # floyd-warshall for all pairs shortest path
# # def floyd_warshall_shortest_path(G, num_nodes):
# #     # 初始化距离矩阵
# #     shortest_path_matrix = np.full((num_nodes, num_nodes), np.inf)
# #     for i in range(num_nodes):
# #         shortest_path_matrix[i, i] = 0
# #     for u, v in G.edges():
# #         shortest_path_matrix[u, v] = 1
# #         shortest_path_matrix[v, u] = 1
# #
# #     # Floyd-Warshall算法核心
# #     for k in range(num_nodes):
# #         for i in range(num_nodes):
# #             for j in range(num_nodes):
# #                 shortest_path_matrix[i, j] = min(
# #                     shortest_path_matrix[i, j],
# #                     shortest_path_matrix[i, k] + shortest_path_matrix[k, j]
# #                 )
# #
# #     return shortest_path_matrix
#
#
# #
# def calculate_calibration_mask(edge_index, num_nodes, lambda_val=0.5):
#     # 创建一个空的图
#     G = nx.Graph()
#     G.add_nodes_from(range(num_nodes))
#     # 添加边
#     edges = edge_index.t().tolist()
#     G.add_edges_from(edges)
#
#     # 初始化最短路径距离矩阵
#     shortest_path_matrix = torch.full((num_nodes, num_nodes), np.inf)
#
#     # 计算最短路径距离
#     for i in range(num_nodes):
#         for j in range(num_nodes):
#             if nx.has_path(G, i, j):
#                 shortest_path_matrix[i, j] = nx.shortest_path_length(G, i, j)
#
#     # 从最短距离为3之后开始指数衰减
#     # 最短距离小于3的设为1
#     calibration_mask = torch.ones((num_nodes, num_nodes))
#
#     # 处理最短距离大于等于3x的情况
#     mask = shortest_path_matrix >= 3
#     # 最短距离超过10的统一按10处理
#     adjusted_shortest_path = torch.clamp(shortest_path_matrix[mask], max=10)
#     calibration_mask[mask] = lambda_val ** (adjusted_shortest_path - 2)
#
#     return calibration_mask
#
#
#
#


import torch
import numpy as np
import networkx as nx


def calculate_calibration_mask(edge_index, num_nodes, lambda_val=0.5):
    # 创建一个空的图
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    # 添加边
    edges = edge_index.t().tolist()
    G.add_edges_from(edges)

    # 使用 Floyd - Warshall 算法计算最短路径矩阵
    shortest_path_matrix = torch.tensor(nx.floyd_warshall_numpy(G), dtype=torch.float32)

    # 从最短距离为 3 之后开始指数衰减
    # 最短距离小于 3 的设为 1
    calibration_mask = torch.ones((num_nodes, num_nodes))

    # 处理最短距离大于等于 3 的情况
    mask = shortest_path_matrix >= 3
    # 最短距离超过 10 的统一按 10 处理
    adjusted_shortest_path = torch.clamp(shortest_path_matrix[mask], max=10)
    calibration_mask[mask] = lambda_val ** (adjusted_shortest_path - 2)

    return calibration_mask

