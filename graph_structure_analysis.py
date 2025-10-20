#!/usr/bin/env python3
"""
图结构分析示例

演示特征矩阵、邻接列表和节点高度之间的对应关系
"""

import torch
import numpy as np

def analyze_graph_structure():
    """
    分析图结构信息的对应关系
    """
    print("=" * 60)
    print("图结构分析")
    print("=" * 60)
    
    # 模拟你提到的图结构数据
    num_nodes = 5
    feature_dim = 1165
    
    # 特征矩阵 (每个节点一行特征)
    features = torch.randn(num_nodes, feature_dim)  # 随机生成示例特征
    
    # 邻接列表 (边的起点和终点)
    adjacency_list = torch.tensor([[0, 1],
                                   [1, 2],
                                   [1, 3],
                                   [3, 4]])
    
    # 节点高度
    heights = torch.tensor([3, 2, 0, 1, 0])
    
    print(f"1. 特征矩阵分析:")
    print(f"   - 形状: {features.shape}")
    print(f"   - 节点数量: {features.shape[0]}")
    print(f"   - 每个节点特征维度: {features.shape[1]}")
    print()
    
    print(f"2. 节点-特征对应关系:")
    for i in range(num_nodes):
        print(f"   - 节点{i}: features[{i}, :] -> {feature_dim}维特征向量")
        print(f"     特征向量前5维示例: {features[i, :5]}")
    print()
    
    print(f"3. 邻接列表分析:")
    print(f"   - 边数量: {len(adjacency_list)}")
    print(f"   - 边的连接关系:")
    for i, edge in enumerate(adjacency_list):
        start, end = edge[0].item(), edge[1].item()
        print(f"     边{i}: 节点{start} -> 节点{end}")
    print()
    
    print(f"4. 树结构可视化:")
    print("   节点0 (高度3, 根节点)")
    print("     └── 节点1 (高度2)")
    print("         ├── 节点2 (高度0, 叶子)")
    print("         └── 节点3 (高度1)")
    print("             └── 节点4 (高度0, 叶子)")
    print()
    
    print(f"5. 节点高度分析:")
    for i, height in enumerate(heights):
        node_type = "根节点" if height == heights.max() else ("叶子节点" if height == 0 else "中间节点")
        print(f"   - 节点{i}: 高度{height.item()} ({node_type})")
    print()
    
    # 验证一致性
    print(f"6. 一致性验证:")
    print(f"   - 特征矩阵节点数 == 高度向量长度: {features.shape[0] == len(heights)}")
    print(f"   - 所有边的节点ID都在有效范围内: {adjacency_list.max() < num_nodes}")
    print(f"   - 边数量 == 节点数-1 (树结构): {len(adjacency_list) == num_nodes - 1}")

def create_adjacency_matrix(adjacency_list, num_nodes):
    """
    从邻接列表创建邻接矩阵
    """
    adj_matrix = torch.zeros(num_nodes, num_nodes, dtype=torch.bool)
    for edge in adjacency_list:
        start, end = edge[0].item(), edge[1].item()
        adj_matrix[start, end] = True
    return adj_matrix

def analyze_feature_composition():
    """
    分析1165维特征向量的组成
    """
    print("\n" + "=" * 60)
    print("特征向量组成分析 (1165维)")
    print("=" * 60)
    
    feature_composition = [
        ("节点类型和连接类型", 2),
        ("过滤条件", 9),
        ("过滤掩码", 3),
        ("直方图特征", 147),  # 3 * 49
        ("表ID", 1),
        ("表采样位图", 1000),
        ("其他/填充", 3)  # 1165 - (2+9+3+147+1+1000) = 3
    ]
    
    total_dims = 0
    for name, dims in feature_composition:
        print(f"   - {name}: {dims}维")
        total_dims += dims
    
    print(f"\n   总计: {total_dims}维")
    
    # 显示特征向量的分段
    print(f"\n特征向量分段索引:")
    start_idx = 0
    for name, dims in feature_composition:
        end_idx = start_idx + dims
        print(f"   - {name}: [{start_idx}:{end_idx}]")
        start_idx = end_idx

if __name__ == "__main__":
    analyze_graph_structure()
    analyze_feature_composition()
    
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("1. 特征矩阵的第一维确实等于节点数量")
    print("2. 每个节点对应特征矩阵中的一行")
    print("3. 邻接列表定义了节点之间的连接关系")
    print("4. 节点高度反映了在树结构中的层次位置")
    print("5. 这种设计使得图神经网络能够:")
    print("   - 通过特征矩阵获取每个节点的特征")
    print("   - 通过邻接信息进行消息传递")
    print("   - 通过高度信息理解层次结构")
