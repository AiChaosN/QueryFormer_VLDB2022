#!/usr/bin/env python3
"""
演示方法使用示例

展示如何使用新添加的演示方法来理解数据处理流程
"""

# 使用示例1: 演示数据集中的真实查询
def example_real_data():
    """
    使用数据集中的真实数据进行演示
    """
    # 假设你已经创建了dataset对象
    # dataset = PlanTreeDataset(...)
    
    # 演示第0个查询
    result = dataset.demo_with_sample_data(0)
    
    # 结果包含：
    # - tree_root: 树结构的根节点
    # - tree_dict: 图结构字典
    # - collated_dict: 模型输入格式
    # - original_data: 原始JSON数据
    
    return result

# 使用示例2: 演示自定义JSON
def example_custom_json():
    """
    使用自定义JSON进行演示
    """
    custom_json = """{
        "Plan": {
            "Node Type": "Seq Scan",
            "Relation Name": "users",
            "Alias": "u",
            "Filter": "(age > 25)",
            "Actual Rows": 1000,
            "Total Cost": 100.0
        },
        "Execution Time": 50.0
    }"""
    
    # 演示自定义JSON的处理
    result = dataset.demo_complete_pipeline(custom_json, query_id=123)
    return result

# 使用示例3: 只演示特定步骤
def example_specific_steps():
    """
    只演示特定的处理步骤
    """
    json_string = '{"Plan": {...}, "Execution Time": 100}'
    
    # 步骤1: 只演示JSON到树的转换
    tree_result = dataset.demo_json_to_tree_conversion(json_string)
    
    if tree_result:
        tree_root = tree_result['tree_root']
        
        # 步骤2: 只演示树结构可视化
        dataset.demo_tree_structure_visualization(tree_root)
        
        # 步骤3: 只演示特征编码
        dataset.demo_feature_encoding(tree_root)

if __name__ == "__main__":
    print("演示方法使用示例")
    print("请在创建dataset对象后调用相应的演示方法")
    print("详细使用方法请参考 Demo_Data_Processing.ipynb")
