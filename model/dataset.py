import torch
from torch.utils.data import Dataset
import numpy as np
import json
import pandas as pd
import sys, os
from collections import deque
from .database_util import formatFilter, formatJoin, TreeNode, filterDict2Hist
from .database_util import *

class PlanTreeDataset(Dataset):
    def __init__(self, json_df : pd.DataFrame, train : pd.DataFrame, encoding, hist_file, card_norm, cost_norm, to_predict, table_sample):

        self.json_df = json_df  # 保存原始数据框用于演示
        self.table_sample = table_sample
        self.encoding = encoding
        self.hist_file = hist_file
        
        self.length = len(json_df)
        # train = train.loc[json_df['id']]
        
        nodes = [json.loads(plan)['Plan'] for plan in json_df['json']]
        self.cards = [node['Actual Rows'] for node in nodes]
        self.costs = [json.loads(plan)['Execution Time'] for plan in json_df['json']]
        
        self.card_labels = torch.from_numpy(card_norm.normalize_labels(self.cards))
        self.cost_labels = torch.from_numpy(cost_norm.normalize_labels(self.costs))
        
        self.to_predict = to_predict
        if to_predict == 'cost':
            self.gts = self.costs
            self.labels = self.cost_labels
        elif to_predict == 'card':
            self.gts = self.cards
            self.labels = self.card_labels
        elif to_predict == 'both': ## try not to use, just in case
            self.gts = self.costs
            self.labels = self.cost_labels
        else:
            raise Exception('Unknown to_predict type')
            
        idxs = list(json_df['id'])
        
    
        self.treeNodes = [] ## for mem collection
        self.collated_dicts = [self.js_node2dict(i,node) for i,node in zip(idxs, nodes)]

    def js_node2dict(self, idx, node):
        treeNode = self.traversePlan(node, idx, self.encoding)
        _dict = self.node2dict(treeNode)
        collated_dict = self.pre_collate(_dict)
        
        self.treeNodes.clear()
        del self.treeNodes[:]

        return collated_dict

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        
        return self.collated_dicts[idx], (self.cost_labels[idx], self.card_labels[idx])

    def old_getitem(self, idx):
        return self.dicts[idx], (self.cost_labels[idx], self.card_labels[idx])
      
    ## pre-process first half of old collator
    def pre_collate(self, the_dict, max_node = 30, rel_pos_max = 20):

        x = pad_2d_unsqueeze(the_dict['features'], max_node)
        N = len(the_dict['features'])
        attn_bias = torch.zeros([N+1,N+1], dtype=torch.float)
        
        edge_index = the_dict['adjacency_list'].t()
        if len(edge_index) == 0:
            shortest_path_result = np.array([[0]])
            path = np.array([[0]])
            adj = torch.tensor([[0]]).bool()
        else:
            adj = torch.zeros([N,N], dtype=torch.bool)
            adj[edge_index[0,:], edge_index[1,:]] = True
            
            shortest_path_result = floyd_warshall_rewrite(adj.numpy())
        
        rel_pos = torch.from_numpy((shortest_path_result)).long()

        
        attn_bias[1:, 1:][rel_pos >= rel_pos_max] = float('-inf')
        
        attn_bias = pad_attn_bias_unsqueeze(attn_bias, max_node + 1)
        rel_pos = pad_rel_pos_unsqueeze(rel_pos, max_node)

        heights = pad_1d_unsqueeze(the_dict['heights'], max_node)
        
        return {
            'x' : x,
            'attn_bias': attn_bias,
            'rel_pos': rel_pos,
            'heights': heights
        }


    def node2dict(self, treeNode):

        adj_list, num_child, features = self.topo_sort(treeNode)
        heights = self.calculate_height(adj_list, len(features))

        return {
            'features' : torch.FloatTensor(np.array(features)),
            'heights' : torch.LongTensor(heights),
            'adjacency_list' : torch.LongTensor(np.array(adj_list)),
          
        }
    
    def topo_sort(self, root_node):
#        nodes = []
        adj_list = [] #from parent to children
        num_child = []
        features = []

        toVisit = deque()
        toVisit.append((0,root_node))
        next_id = 1
        while toVisit:
            idx, node = toVisit.popleft()
#            nodes.append(node)
            features.append(node.feature)
            num_child.append(len(node.children))
            for child in node.children:
                toVisit.append((next_id,child))
                adj_list.append((idx,next_id))
                next_id += 1
        
        return adj_list, num_child, features
    
    # 遍历计划树，生成树节点
    def traversePlan(self, plan, idx, encoding): # bfs accumulate plan

        nodeType = plan['Node Type']
        typeId = encoding.encode_type(nodeType)
        card = None #plan['Actual Rows']
        filters, alias = formatFilter(plan)
        join = formatJoin(plan)
        joinId = encoding.encode_join(join)
        filters_encoded = encoding.encode_filters(filters, alias)
        
        root = TreeNode(nodeType, typeId, filters, card, joinId, join, filters_encoded)
        
        self.treeNodes.append(root)

        if 'Relation Name' in plan:
            root.table = plan['Relation Name']
            root.table_id = encoding.encode_table(plan['Relation Name'])
        root.query_id = idx
        
        root.feature = node2feature(root, encoding, self.hist_file, self.table_sample)
        #    print(root)
        if 'Plans' in plan:
            for subplan in plan['Plans']:
                subplan['parent'] = plan
                node = self.traversePlan(subplan, idx, encoding)
                node.parent = root
                root.addChild(node)
        return root

    def calculate_height(self, adj_list,tree_size):
        if tree_size == 1:
            return np.array([0])

        adj_list = np.array(adj_list)
        node_ids = np.arange(tree_size, dtype=int)
        node_order = np.zeros(tree_size, dtype=int)
        uneval_nodes = np.ones(tree_size, dtype=bool)

        parent_nodes = adj_list[:,0]
        child_nodes = adj_list[:,1]

        n = 0
        while uneval_nodes.any():
            uneval_mask = uneval_nodes[child_nodes]
            unready_parents = parent_nodes[uneval_mask]

            node2eval = uneval_nodes & ~np.isin(node_ids, unready_parents)
            node_order[node2eval] = n
            uneval_nodes[node2eval] = False
            n += 1
        return node_order 

    # ==================== 演示方法 ====================
    
    def demo_json_to_tree_conversion(self, json_string, query_id=0):
        """
        演示JSON执行计划到树结构的转换过程
        
        Args:
            json_string: JSON格式的执行计划字符串
            query_id: 查询ID
            
        Returns:
            dict: 包含原始JSON、解析后的计划和树结构的字典
        """
        print("=" * 80)
        print("JSON到树结构转换演示")
        print("=" * 80)
        
        # 1. 显示原始JSON
        print("\n1. 原始JSON执行计划:")
        print("-" * 40)
        try:
            # 格式化显示JSON
            parsed_json = json.loads(json_string)
            print(json.dumps(parsed_json, indent=2, ensure_ascii=False)[:1000] + "...")
            
            # 2. 提取Plan部分
            plan = parsed_json['Plan']
            execution_time = parsed_json.get('Execution Time', 0)
            
            print(f"\n2. 提取的关键信息:")
            print("-" * 40)
            print(f"执行时间: {execution_time} ms")
            print(f"根节点类型: {plan['Node Type']}")
            print(f"实际行数: {plan.get('Actual Rows', 'N/A')}")
            print(f"总成本: {plan.get('Total Cost', 'N/A')}")
            
            # 3. 转换为树结构
            print(f"\n3. 转换为树结构:")
            print("-" * 40)
            tree_root = self.traversePlan(plan, query_id, self.encoding)
            
            # 显示树结构
            print("树结构层次:")
            TreeNode.print_nested(tree_root)
            
            return {
                'original_json': parsed_json,
                'execution_time': execution_time,
                'plan': plan,
                'tree_root': tree_root
            }
            
        except Exception as e:
            print(f"转换过程中出现错误: {e}")
            return None

    def demo_tree_structure_visualization(self, tree_root):
        """
        演示树结构的详细可视化
        
        Args:
            tree_root: TreeNode对象，树的根节点
        """
        print("\n" + "=" * 80)
        print("树结构详细可视化")
        print("=" * 80)
        
        def analyze_node(node, level=0):
            indent = "  " * level
            print(f"{indent}节点 {level}:")
            print(f"{indent}  - 类型: {node.nodeType} (ID: {node.typeId})")
            print(f"{indent}  - 表: {node.table} (ID: {node.table_id})")
            print(f"{indent}  - 连接: {node.join_str}")
            print(f"{indent}  - 过滤条件: {node.filter}")
            print(f"{indent}  - 过滤字典: {node.filterDict}")
            print(f"{indent}  - 子节点数量: {len(node.children)}")
            
            if hasattr(node, 'feature') and node.feature is not None:
                print(f"{indent}  - 特征向量维度: {len(node.feature)}")
                print(f"{indent}  - 特征向量前10维: {node.feature[:10]}")
            
            print()
            
            for i, child in enumerate(node.children):
                print(f"{indent}子节点 {i+1}:")
                analyze_node(child, level + 1)
        
        analyze_node(tree_root)

    def demo_feature_encoding(self, tree_root):
        """
        演示特征编码过程
        
        Args:
            tree_root: TreeNode对象，树的根节点
        """
        print("\n" + "=" * 80)
        print("特征编码演示")
        print("=" * 80)
        
        def show_node_features(node, level=0):
            indent = "  " * level
            print(f"{indent}节点 {level} 特征编码:")
            print(f"{indent}" + "-" * 30)
            
            if hasattr(node, 'feature') and node.feature is not None:
                feature = node.feature
                print(f"{indent}总特征维度: {len(feature)}")
                
                # 分解特征向量
                idx = 0
                
                # 1. 节点类型和连接类型 (2维)
                type_join = feature[idx:idx+2]
                idx += 2
                print(f"{indent}1. 类型&连接特征 [0:2]: {type_join}")
                print(f"{indent}   - 节点类型ID: {type_join[0]}")
                print(f"{indent}   - 连接ID: {type_join[1]}")
                
                # 2. 过滤条件 (9维)
                filters = feature[idx:idx+9].reshape(3, 3)
                idx += 9
                print(f"{indent}2. 过滤条件特征 [2:11]: ")
                for i, filt in enumerate(filters):
                    print(f"{indent}   - 过滤条件{i+1}: 列ID={filt[0]}, 操作符ID={filt[1]}, 值={filt[2]}")
                
                # 3. 过滤条件掩码 (3维)
                mask = feature[idx:idx+3]
                idx += 3
                print(f"{indent}3. 过滤掩码 [11:14]: {mask}")
                
                # 4. 直方图特征 (147维)
                hist_size = 147  # 3 * 49
                hists = feature[idx:idx+hist_size]
                idx += hist_size
                print(f"{indent}4. 直方图特征 [14:{14+hist_size}]: 维度={len(hists)}")
                print(f"{indent}   - 前5维: {hists[:5]}")
                print(f"{indent}   - 后5维: {hists[-5:]}")
                
                # 5. 表ID (1维)
                table_id = feature[idx:idx+1]
                idx += 1
                print(f"{indent}5. 表ID [{14+hist_size}:{14+hist_size+1}]: {table_id}")
                
                # 6. 表采样位图 (1000维)
                sample = feature[idx:idx+1000]
                idx += 1000
                print(f"{indent}6. 表采样位图 [{14+hist_size+1}:{14+hist_size+1001}]: 维度={len(sample)}")
                print(f"{indent}   - 非零元素数量: {np.count_nonzero(sample)}")
                print(f"{indent}   - 前5维: {sample[:5]}")
                
                print(f"{indent}实际使用的特征维度: {idx}")
                print()
            else:
                print(f"{indent}该节点没有特征向量")
                print()
            
            for i, child in enumerate(node.children):
                show_node_features(child, level + 1)
        
        show_node_features(tree_root)

    def demo_complete_pipeline(self, json_string, query_id=0):
        """
        演示完整的数据处理流水线
        
        Args:
            json_string: JSON格式的执行计划字符串
            query_id: 查询ID
        """
        print("=" * 100)
        print("完整数据处理流水线演示")
        print("=" * 100)
        
        # 步骤1: JSON到树转换
        result = self.demo_json_to_tree_conversion(json_string, query_id)
        if result is None:
            return
        
        tree_root = result['tree_root']
        
        # 步骤2: 树结构可视化
        self.demo_tree_structure_visualization(tree_root)
        
        # 步骤3: 特征编码
        self.demo_feature_encoding(tree_root)
        
        # 步骤4: 转换为图结构
        print("\n" + "=" * 80)
        print("图结构转换")
        print("=" * 80)
        
        try:
            # 转换为字典格式
            tree_dict = self.node2dict(tree_root)
            print(f"图结构信息:")
            print(f"  - 节点数量: {len(tree_dict['features'])}")
            print(f"  - 边数量: {len(tree_dict['adjacency_list'])}")
            print(f"  - 特征矩阵形状: {tree_dict['features'].shape}")
            print(f"  - 邻接列表: {tree_dict['adjacency_list']}")
            print(f"  - 节点高度: {tree_dict['heights']}")
            
            # 预处理为模型输入格式
            collated_dict = self.pre_collate(tree_dict)
            print(f"\n预处理后的模型输入:")
            print(f"  - x (特征矩阵): {collated_dict['x'].shape}")
            print(f"  - attn_bias (注意力偏置): {collated_dict['attn_bias'].shape}")
            print(f"  - rel_pos (相对位置): {collated_dict['rel_pos'].shape}")
            print(f"  - heights (高度编码): {collated_dict['heights'].shape}")
            
            return {
                'tree_root': tree_root,
                'tree_dict': tree_dict,
                'collated_dict': collated_dict,
                'original_data': result
            }
            
        except Exception as e:
            print(f"图结构转换过程中出现错误: {e}")
            return None

    def demo_with_sample_data(self, sample_index=0):
        """
        使用数据集中的样本数据进行演示
        
        Args:
            sample_index: 样本索引
        """
        if not hasattr(self, 'json_df') or self.json_df is None:
            print("数据集为空，无法进行演示")
            return
        
        if sample_index >= len(self.json_df):
            print(f"样本索引 {sample_index} 超出数据集大小 {len(self.json_df)}")
            return
        
        # 获取样本数据
        sample_row = self.json_df.iloc[sample_index]
        json_string = sample_row['json']
        query_id = sample_row['id']
        
        print(f"使用数据集中的第 {sample_index} 个样本进行演示")
        print(f"查询ID: {query_id}")
        
        return self.demo_complete_pipeline(json_string, query_id)


def node2feature(node, encoding, hist_file, table_sample):
    # type, join, filter123, mask123
    # 1, 1, 3x3 (9), 3
    # TODO: add sample (or so-called table)
    num_filter = len(node.filterDict['colId'])
    pad = np.zeros((3,3-num_filter))
    filts = np.array(list(node.filterDict.values())) #cols, ops, vals
    ## 3x3 -> 9, get back with reshape 3,3
    filts = np.concatenate((filts, pad), axis=1).flatten() 
    mask = np.zeros(3)
    mask[:num_filter] = 1
    type_join = np.array([node.typeId, node.join])
    
    hists = filterDict2Hist(hist_file, node.filterDict, encoding)


    # table, bitmap, 1 + 1000 bits
    table = np.array([node.table_id])
    if node.table_id == 0:
        sample = np.zeros(1000)
    else:
        sample = table_sample[node.query_id][node.table]
    
    #return np.concatenate((type_join,filts,mask))
    return np.concatenate((type_join, filts, mask, hists, table, sample))
