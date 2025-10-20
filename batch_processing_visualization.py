#!/usr/bin/env python3
"""
批处理可视化示例

直观展示QueryFormer如何处理不同长度的查询批次
"""

import torch
import numpy as np

def visualize_batch_processing():
    """
    可视化批处理过程
    """
    print("=" * 100)
    print("QueryFormer批处理可视化")
    print("=" * 100)
    
    # 模拟一个批次的查询
    batch_queries = [
        {"id": "Q1", "nodes": 3, "type": "简单查询 (Seq Scan)"},
        {"id": "Q2", "nodes": 5, "type": "Hash Join查询"},
        {"id": "Q3", "nodes": 8, "type": "复杂嵌套查询"},
        {"id": "Q4", "nodes": 2, "type": "最简单查询"}
    ]
    
    max_nodes = 10  # 简化示例，实际是30
    batch_size = len(batch_queries)
    
    print(f"批次信息:")
    print(f"- 批次大小: {batch_size}")
    print(f"- 最大节点数: {max_nodes}")
    print(f"- 特征维度: 1165")
    print()
    
    print("原始查询长度:")
    print("-" * 50)
    for query in batch_queries:
        print(f"{query['id']}: {query['nodes']}节点 - {query['type']}")
    print()
    
    # 创建填充后的特征矩阵
    print("填充后的批次特征矩阵:")
    print("-" * 50)
    print(f"形状: [{batch_size}, {max_nodes}, 1165]")
    print()
    
    # 可视化填充过程
    print("填充可视化 (X=有效节点, O=填充节点):")
    print("-" * 50)
    print("查询ID  " + "".join([f"{i:2d}" for i in range(max_nodes)]))
    print("        " + "".join(["--" for _ in range(max_nodes)]))
    
    for i, query in enumerate(batch_queries):
        nodes = query['nodes']
        visualization = ""
        for j in range(max_nodes):
            if j < nodes:
                visualization += " X"
            else:
                visualization += " O"
        print(f"{query['id']:6s}  {visualization}")
    print()

def visualize_attention_mask():
    """
    可视化注意力掩码
    """
    print("注意力掩码可视化:")
    print("-" * 50)
    
    # 以5节点查询为例
    actual_nodes = 5
    max_nodes = 8  # 简化示例
    total_nodes = max_nodes + 1  # +1 for super token
    
    print(f"示例: 5节点查询，填充到8节点")
    print(f"注意力矩阵大小: {total_nodes} x {total_nodes} (包含Super Token)")
    print()
    
    # 创建注意力掩码
    mask = torch.full((total_nodes, total_nodes), float('-inf'))
    
    # Super token (index 0) can attend to all valid nodes
    mask[0, :actual_nodes+1] = 0
    mask[:actual_nodes+1, 0] = 0
    
    # Valid nodes can attend to each other
    mask[1:actual_nodes+1, 1:actual_nodes+1] = 0
    
    print("注意力掩码矩阵 (0=可以注意, -∞=被掩码):")
    print("行/列索引: S=Super Token, 1-5=有效节点, 6-8=填充节点")
    print("        S  1  2  3  4  5  6  7  8")
    
    labels = ['S'] + [str(i) for i in range(1, total_nodes)]
    for i, label in enumerate(labels):
        row_str = f"    {label}   "
        for j in range(total_nodes):
            if mask[i, j] == 0:
                row_str += " 0 "
            else:
                row_str += "-∞ "
        print(row_str)
    print()

def visualize_forward_pass():
    """
    可视化前向传播过程
    """
    print("前向传播过程可视化:")
    print("-" * 50)
    
    batch_size = 2
    actual_nodes = [3, 5]  # 两个查询分别有3和5个节点
    max_nodes = 8
    hidden_dim = 169  # 简化的hidden_dim
    
    print(f"批次: {batch_size}个查询")
    print(f"查询1: {actual_nodes[0]}节点, 查询2: {actual_nodes[1]}节点")
    print(f"填充到: {max_nodes}节点")
    print()
    
    # Step 1: 输入特征
    print("Step 1: 输入特征矩阵")
    print(f"  形状: [{batch_size}, {max_nodes}, 1165]")
    print("  内容: 原始特征向量 + 填充零向量")
    print()
    
    # Step 2: 特征嵌入
    print("Step 2: 特征嵌入")
    print(f"  FeatureEmbed: [{batch_size}, {max_nodes}, 1165] -> [{batch_size}, {max_nodes}, {hidden_dim}]")
    print("  作用: 将原始特征转换为模型内部表示")
    print()
    
    # Step 3: 添加位置编码
    print("Step 3: 添加高度编码")
    print(f"  + height_encoding: [{batch_size}, {max_nodes}, {hidden_dim}]")
    print("  作用: 添加树结构的层次信息")
    print()
    
    # Step 4: 添加Super Token
    print("Step 4: 添加Super Token")
    print(f"  concat: [{batch_size}, 1, {hidden_dim}] + [{batch_size}, {max_nodes}, {hidden_dim}]")
    print(f"  结果: [{batch_size}, {max_nodes+1}, {hidden_dim}]")
    print("  作用: Super Token用于聚合全局信息")
    print()
    
    # Step 5: Transformer处理
    print("Step 5: Transformer层处理")
    print(f"  输入: [{batch_size}, {max_nodes+1}, {hidden_dim}]")
    print("  处理: 多头自注意力 + 前馈网络 (重复8层)")
    print(f"  输出: [{batch_size}, {max_nodes+1}, {hidden_dim}]")
    print("  掩码: 确保只有有效节点参与计算")
    print()
    
    # Step 6: 预测
    print("Step 6: 最终预测")
    print(f"  取Super Token: output[:, 0, :] -> [{batch_size}, {hidden_dim}]")
    print(f"  预测层: [{batch_size}, {hidden_dim}] -> [{batch_size}, 1]")
    print("  输出: 查询成本/基数预测")
    print()

def demonstrate_efficiency():
    """
    演示效率优势
    """
    print("效率优势演示:")
    print("-" * 50)
    
    print("1. 并行处理优势:")
    print("   顺序处理:")
    print("   - 查询1 (3节点) -> 处理时间: T1")
    print("   - 查询2 (5节点) -> 处理时间: T2") 
    print("   - 查询3 (8节点) -> 处理时间: T3")
    print("   - 总时间: T1 + T2 + T3")
    print()
    
    print("   批处理 (填充):")
    print("   - 所有查询同时处理 -> 处理时间: max(T1, T2, T3)")
    print("   - GPU并行计算 -> 显著提速")
    print()
    
    print("2. 内存vs计算权衡:")
    print("   内存开销:")
    print("   - 填充节点占用额外内存")
    print("   - 但支持批处理，提高吞吐量")
    print()
    
    print("   计算优化:")
    print("   - 注意力掩码屏蔽填充节点")
    print("   - 只有有效节点参与实际计算")
    print("   - 矩阵运算高度优化")
    print()

if __name__ == "__main__":
    visualize_batch_processing()
    visualize_attention_mask()
    visualize_forward_pass()
    demonstrate_efficiency()
    
    print("=" * 100)
    print("关键要点总结")
    print("=" * 100)
    print("1. QueryFormer不会根据节点数量循环不同次数")
    print("2. 所有查询都填充到固定长度 (max_nodes=30)")
    print("3. 使用注意力掩码区分有效节点和填充节点")
    print("4. Transformer层数是固定的 (默认8层)")
    print("5. Super Token聚合所有信息进行最终预测")
    print("6. 这种设计实现了高效的并行处理")
