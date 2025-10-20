#!/usr/bin/env python3
"""
数据处理演示脚本

本脚本演示QueryFormer项目中JSON执行计划到树结构转换以及特征编码的完整过程
"""

import sys
import os
import pandas as pd
import torch
import json

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.dataset import PlanTreeDataset
from model.database_util import get_hist_file, get_job_table_sample, Encoding, Normalizer

def setup_demo_environment():
    """
    设置演示环境，加载必要的数据和配置
    """
    print("正在设置演示环境...")
    
    # 数据路径
    data_path = './data/imdb/'
    
    # 加载训练数据
    print("加载训练数据...")
    train_df = pd.read_csv(data_path + 'plan_and_cost/train_plan_part0.csv').head(10)  # 只取前10条数据用于演示
    print(f"加载了 {len(train_df)} 条训练数据")
    
    # 加载直方图文件
    print("加载直方图文件...")
    hist_file = get_hist_file(data_path + 'histogram_string.csv')
    print(f"加载了 {len(hist_file)} 个直方图")
    
    # 加载表采样数据
    print("加载表采样数据...")
    table_sample = get_job_table_sample(data_path + 'train')
    print(f"加载了 {len(table_sample)} 个查询的表采样数据")
    
    # 创建编码器
    print("创建编码器...")
    # 从checkpoints加载编码器或创建新的
    try:
        encoding_ckpt = torch.load('./checkpoints/encoding.pt', map_location='cpu')
        encoding = encoding_ckpt['encoding']
        print("从checkpoint加载编码器")
    except:
        # 如果没有checkpoint，创建基本的编码器
        print("创建新的编码器")
        column_min_max_vals = {}
        col2idx = {'NA': 0}
        encoding = Encoding(column_min_max_vals, col2idx)
    
    # 创建标准化器
    cost_norm = Normalizer(-3.61192, 12.290855)
    card_norm = Normalizer(1, 100)
    
    return train_df, hist_file, table_sample, encoding, cost_norm, card_norm

def demo_single_query(dataset, query_index=0):
    """
    演示单个查询的处理过程
    """
    print(f"\n{'='*100}")
    print(f"演示第 {query_index} 个查询的处理过程")
    print(f"{'='*100}")
    
    # 使用数据集中的样本数据进行演示
    return dataset.demo_with_sample_data(query_index)

def demo_custom_json(dataset):
    """
    演示自定义JSON的处理过程
    """
    print(f"\n{'='*100}")
    print("演示自定义JSON的处理过程")
    print(f"{'='*100}")
    
    # 创建一个简单的示例JSON
    sample_json = """{
        "Plan": {
            "Node Type": "Seq Scan",
            "Parallel Aware": false,
            "Relation Name": "title",
            "Alias": "t",
            "Startup Cost": 0.0,
            "Total Cost": 67602.3,
            "Plan Rows": 1116092,
            "Plan Width": 94,
            "Actual Startup Time": 0.035,
            "Actual Total Time": 322.837,
            "Actual Rows": 1107925,
            "Actual Loops": 1,
            "Filter": "(production_year > 2004)",
            "Rows Removed by Filter": 1420387
        },
        "Planning Time": 1.7,
        "Triggers": [],
        "Execution Time": 349.797
    }"""
    
    return dataset.demo_complete_pipeline(sample_json, query_id=999)

def demo_complex_query(dataset):
    """
    演示复杂查询的处理过程（包含连接的查询）
    """
    print(f"\n{'='*100}")
    print("演示复杂查询的处理过程")
    print(f"{'='*100}")
    
    # 创建一个包含连接的复杂查询示例
    complex_json = """{
        "Plan": {
            "Node Type": "Hash Join",
            "Parallel Aware": false,
            "Join Type": "Inner",
            "Startup Cost": 22540.58,
            "Total Cost": 96783.45,
            "Plan Rows": 236523,
            "Plan Width": 119,
            "Actual Startup Time": 369.985,
            "Actual Total Time": 518.487,
            "Actual Rows": 94604,
            "Actual Loops": 1,
            "Inner Unique": false,
            "Hash Cond": "(t.id = mi_idx.movie_id)",
            "Plans": [
                {
                    "Node Type": "Seq Scan",
                    "Parent Relationship": "Outer",
                    "Parallel Aware": true,
                    "Relation Name": "title",
                    "Alias": "t",
                    "Startup Cost": 0.0,
                    "Total Cost": 49166.46,
                    "Plan Rows": 649574,
                    "Plan Width": 94,
                    "Actual Startup Time": 0.366,
                    "Actual Total Time": 147.047,
                    "Actual Rows": 514421,
                    "Actual Loops": 1,
                    "Filter": "(kind_id = 7)",
                    "Rows Removed by Filter": 328349
                },
                {
                    "Node Type": "Hash",
                    "Parent Relationship": "Inner",
                    "Parallel Aware": true,
                    "Startup Cost": 15122.68,
                    "Total Cost": 15122.68,
                    "Plan Rows": 383592,
                    "Plan Width": 25,
                    "Actual Startup Time": 103.547,
                    "Actual Total Time": 103.547,
                    "Actual Rows": 306703,
                    "Actual Loops": 1,
                    "Plans": [
                        {
                            "Node Type": "Seq Scan",
                            "Parent Relationship": "Outer",
                            "Parallel Aware": true,
                            "Relation Name": "movie_info_idx",
                            "Alias": "mi_idx",
                            "Startup Cost": 0.0,
                            "Total Cost": 15122.68,
                            "Plan Rows": 383592,
                            "Plan Width": 25,
                            "Actual Startup Time": 0.28,
                            "Actual Total Time": 54.382,
                            "Actual Rows": 306703,
                            "Actual Loops": 1,
                            "Filter": "(info_type_id > 99)",
                            "Rows Removed by Filter": 153308
                        }
                    ]
                }
            ]
        },
        "Planning Time": 2.382,
        "Triggers": [],
        "Execution Time": 654.241
    }"""
    
    return dataset.demo_complete_pipeline(complex_json, query_id=1000)

def main():
    """
    主演示函数
    """
    print("QueryFormer数据处理流程演示")
    print("="*50)
    
    try:
        # 设置演示环境
        train_df, hist_file, table_sample, encoding, cost_norm, card_norm = setup_demo_environment()
        
        # 创建数据集
        print("\n创建数据集...")
        dataset = PlanTreeDataset(
            json_df=train_df,
            train=None,
            encoding=encoding,
            hist_file=hist_file,
            card_norm=card_norm,
            cost_norm=cost_norm,
            to_predict='cost',
            table_sample=table_sample
        )
        print("数据集创建完成")
        
        # 演示1: 使用数据集中的真实数据
        print("\n" + "="*100)
        print("演示1: 使用数据集中的真实查询数据")
        print("="*100)
        result1 = demo_single_query(dataset, 0)
        
        # 演示2: 使用自定义的简单JSON
        print("\n" + "="*100)
        print("演示2: 使用自定义的简单查询")
        print("="*100)
        result2 = demo_custom_json(dataset)
        
        # 演示3: 使用复杂的连接查询
        print("\n" + "="*100)
        print("演示3: 使用复杂的连接查询")
        print("="*100)
        result3 = demo_complex_query(dataset)
        
        print("\n" + "="*100)
        print("演示完成！")
        print("="*100)
        
        print("\n总结:")
        print("- 演示了JSON执行计划到树结构的转换过程")
        print("- 展示了树结构的详细信息和层次关系")
        print("- 分析了特征编码的各个组成部分")
        print("- 演示了图结构转换和模型输入格式")
        print("- 涵盖了简单查询和复杂连接查询的处理")
        
        return True
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
