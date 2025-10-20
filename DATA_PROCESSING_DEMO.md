# QueryFormer 数据处理演示

本文档说明了如何使用新增的演示方法来理解QueryFormer项目中JSON执行计划到树结构转换以及特征编码的完整过程。

## 新增的演示方法

我已经在 `PlanTreeDataset` 类中添加了以下演示方法：

### 1. `demo_json_to_tree_conversion(json_string, query_id=0)`
- **功能**: 演示JSON执行计划到树结构的转换过程
- **输入**: JSON字符串和查询ID
- **输出**: 包含原始JSON、解析后的计划和树结构的字典
- **展示内容**:
  - 原始JSON执行计划（格式化显示）
  - 提取的关键信息（执行时间、节点类型、行数、成本等）
  - 转换后的树结构层次

### 2. `demo_tree_structure_visualization(tree_root)`
- **功能**: 演示树结构的详细可视化
- **输入**: TreeNode对象（树的根节点）
- **展示内容**:
  - 每个节点的详细信息（类型、表、连接、过滤条件等）
  - 节点的特征向量维度和前10维数值
  - 树的层次结构和父子关系

### 3. `demo_feature_encoding(tree_root)`
- **功能**: 演示特征编码过程
- **输入**: TreeNode对象（树的根节点）
- **展示内容**:
  - 每个节点的完整特征向量分解
  - 特征向量各部分的含义和数值
  - 特征编码的详细组成：
    - 节点类型和连接类型 (2维)
    - 过滤条件 (9维)
    - 过滤掩码 (3维)
    - 直方图特征 (147维)
    - 表ID (1维)
    - 表采样位图 (1000维)

### 4. `demo_complete_pipeline(json_string, query_id=0)`
- **功能**: 演示完整的数据处理流水线
- **输入**: JSON字符串和查询ID
- **展示内容**:
  - 完整的处理流程（JSON→树→特征→图结构）
  - 图结构转换信息
  - 模型输入格式的预处理结果

### 5. `demo_with_sample_data(sample_index=0)`
- **功能**: 使用数据集中的样本数据进行演示
- **输入**: 样本索引
- **展示内容**: 调用完整流水线演示数据集中的真实查询

## 使用方法

### 方法1: 使用Jupyter Notebook（推荐）

1. 打开 `Demo_Data_Processing.ipynb`
2. 按顺序运行所有单元格
3. 观察详细的演示输出

### 方法2: 使用Python脚本

```python
# 运行演示脚本
python demo_data_processing.py
```

### 方法3: 在现有代码中使用

```python
# 创建数据集
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

# 演示数据集中的第0个查询
result = dataset.demo_with_sample_data(0)

# 或者演示自定义JSON
custom_json = '{"Plan": {...}, "Execution Time": 100}'
result = dataset.demo_complete_pipeline(custom_json)
```

## 演示内容说明

### JSON执行计划结构
演示会展示PostgreSQL执行计划的JSON结构，包括：
- `Plan`: 执行计划的树形结构
- `Node Type`: 节点类型（如Seq Scan, Hash Join等）
- `Actual Rows`: 实际返回行数
- `Total Cost`: 总成本估计
- `Execution Time`: 实际执行时间

### 树结构转换
展示如何将JSON的嵌套结构转换为TreeNode对象的层次结构：
- 递归处理子计划
- 提取节点信息（类型、表、过滤条件、连接条件）
- 建立父子关系

### 特征编码详解
详细展示1162维特征向量的构成：

1. **结构特征** (2维)
   - 节点类型ID
   - 连接类型ID

2. **过滤条件特征** (12维)
   - 3个过滤条件 × 3个属性（列ID、操作符ID、归一化值）
   - 3维掩码表示有效过滤条件数量

3. **统计特征** (1147维)
   - 直方图特征：3×49=147维
   - 表ID：1维
   - 表采样位图：1000维

### 图结构转换
展示如何将树结构转换为图神经网络输入：
- 邻接列表构建
- 最短路径距离计算
- 注意力偏置矩阵
- 节点高度编码

## 示例输出

运行演示后，你会看到类似以下的输出：

```
================================================================================
JSON到树结构转换演示
================================================================================

1. 原始JSON执行计划:
----------------------------------------
{
  "Plan": {
    "Node Type": "Gather",
    "Startup Cost": 23540.58,
    "Total Cost": 154548.95,
    ...
  },
  "Execution Time": 654.241
}...

2. 提取的关键信息:
----------------------------------------
执行时间: 654.241 ms
根节点类型: Gather
实际行数: 283812
总成本: 154548.95

3. 转换为树结构:
----------------------------------------
树结构层次:
Gather with [], None, 1 childs
--Hash Join with [], t.id = mi_idx.movie_id, 2 childs
----Seq Scan with ['(kind_id = 7)'], None, 0 childs
----Hash with [], None, 1 childs
------Seq Scan with ['(info_type_id > 99)'], None, 0 childs
```

## 文件说明

- `model/dataset.py`: 包含新增的演示方法
- `Demo_Data_Processing.ipynb`: Jupyter notebook演示
- `demo_data_processing.py`: Python脚本演示
- `demo_usage_example.py`: 使用示例代码
- `DATA_PROCESSING_DEMO.md`: 本说明文档

## 注意事项

1. 确保已正确安装所有依赖包
2. 确保数据文件路径正确
3. 如果没有预训练的编码器checkpoint，会创建基本编码器
4. 演示使用的是数据集的前几条记录，避免输出过长

通过这些演示方法，你可以深入理解QueryFormer项目中数据处理的每个环节，从原始JSON到最终的模型输入格式的完整转换过程。
