# %%
import numpy as np
import os
import torch
import torch.nn as nn
import time
import pandas as pd
from scipy.stats import pearsonr

# %%
from model.util import Normalizer
from model.database_util import get_hist_file, get_job_table_sample, collator
from model.model import QueryFormer
from model.database_util import Encoding
from model.dataset import PlanTreeDataset
from model.trainer import eval_workload, train

# %%
data_path = './data/imdb/'

# %%
class Args:
    # bs = 1024
    # SQ: smaller batch size
    bs = 128
    lr = 0.001
    # epochs = 200
    epochs = 100
    clip_size = 50
    embed_size = 64
    pred_hid = 128
    ffn_dim = 128
    head_size = 12
    n_layers = 8
    dropout = 0.1
    sch_decay = 0.6
    device = 'cuda:0'
    newpath = './results/full/cost/'
    to_predict = 'cost'
    dataset = 'IMDB'  # 添加数据集信息用于训练总结
args = Args()

import os
if not os.path.exists(args.newpath):
    os.makedirs(args.newpath)

# %%
hist_file = get_hist_file(data_path + 'histogram_string.csv')
cost_norm = Normalizer(-3.61192, 12.290855)
card_norm = Normalizer(1,100)

# %%
encoding_ckpt = torch.load('checkpoints/encoding.pt')
encoding = encoding_ckpt['encoding']
checkpoint = torch.load('checkpoints/cost_model.pt', map_location='cpu')

# %%
from model.util import seed_everything
seed_everything()

# %%
model = QueryFormer(emb_size = args.embed_size ,ffn_dim = args.ffn_dim, head_size = args.head_size, \
                 dropout = args.dropout, n_layers = args.n_layers, \
                 use_sample = True, use_hist = True, \
                 pred_hid = args.pred_hid
                )

# %%
_ = model.to(args.device)

# 打印模型参数大小
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n=== Model Parameters (QueryFormer) ===")
print(f"Total Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")
print(f"======================================\n")

# === Model Parameters (QueryFormer) ===
# Total Parameters: 4,484,519
# Trainable Parameters: 4,484,519
# ======================================
# %%
to_predict = 'cost'

# %%
imdb_path = './data/imdb/'
dfs = []  # list to hold DataFrames
# SQ: added
for i in range(2):
#for i in range(18):
    file = imdb_path + 'plan_and_cost/train_plan_part{}.csv'.format(i)
    df = pd.read_csv(file)
    dfs.append(df)

full_train_df = pd.concat(dfs)

val_dfs = []  # list to hold DataFrames
for i in range(18,20):
    file = imdb_path + 'plan_and_cost/train_plan_part{}.csv'.format(i)
    df = pd.read_csv(file)
    val_dfs.append(df)

val_df = pd.concat(val_dfs)

# %%
table_sample = get_job_table_sample(imdb_path+'train')

# %%
train_ds = PlanTreeDataset(full_train_df, None, encoding, hist_file, card_norm, cost_norm, to_predict, table_sample)
val_ds = PlanTreeDataset(val_df, None, encoding, hist_file, card_norm, cost_norm, to_predict, table_sample)

# %%
crit = nn.MSELoss()
model, best_path, training_history = train(model, train_ds, val_ds, crit, cost_norm, args)

# 打印训练总结
# 注意：training_history 是一个 DataFrame，要访问列数据
total_time = training_history['time'].sum() if 'time' in training_history.columns else 0.0
print(f"Training finished. Best model: {best_path}. Total Time: {total_time:.2f}s")

# %%


# %%


# %%
methods = {
    'get_sample' : get_job_table_sample,
    'encoding': encoding,
    'cost_norm': cost_norm,
    'hist_file': hist_file,
    'model': model,
    'device': args.device,
    'bs': 512,
}

# %%


# %%


# %%
# 评估job-light工作负载并保存结果
eval_score_jl, ds_jl, workload_results_jl = eval_workload('job-light', methods, save_results=True, save_path=args.newpath)

# %%
# 评估synthetic工作负载并保存结果  
eval_score_syn, ds_syn, workload_results_syn = eval_workload('synthetic', methods, save_results=True, save_path=args.newpath)

# %%
# 打印工作负载比较总结
# print("\n=== 工作负载比较总结 ===")
# print(f"Job-light - Q-median: {eval_score_jl['q_median']:.4f}, Q-mean: {eval_score_jl['q_mean']:.4f}, 查询数: {workload_results_jl['total_queries']}")
# print(f"Synthetic - Q-median: {eval_score_syn['q_median']:.4f}, Q-mean: {eval_score_syn['q_mean']:.4f}, 查询数: {workload_results_syn['total_queries']}")

# 保存工作负载比较结果
workload_comparison = pd.DataFrame([
    {
        'workload': 'job-light',
        'q_median': eval_score_jl['q_median'],
        'q_90': eval_score_jl['q_90'], 
        'q_mean': eval_score_jl['q_mean'],
        'correlation': workload_results_jl['correlation'],
        'total_queries': workload_results_jl['total_queries'],
        'eval_time': workload_results_jl['workload_eval_time']
    },
    {
        'workload': 'synthetic',
        'q_median': eval_score_syn['q_median'],
        'q_90': eval_score_syn['q_90'],
        'q_mean': eval_score_syn['q_mean'], 
        'correlation': workload_results_syn['correlation'],
        'total_queries': workload_results_syn['total_queries'],
        'eval_time': workload_results_syn['workload_eval_time']
    }
])

workload_comparison.to_csv(args.newpath + 'workload_comparison.csv', index=False)

# %%
# 创建最终的QueryFormer性能报告，用于与GNTO等其他模型比较
# print("\n=== 创建最终性能报告 ===")

try:
    # 读取训练总结
    training_summary = pd.read_csv(args.newpath + 'training_summary.csv')
    
    # 创建标准化的性能报告
    final_performance_report = {
        'model_name': 'QueryFormer',
        'dataset': 'IMDB',
        'framework': 'Transformer-based',
        'training_config': f"epochs={args.epochs}, bs={args.bs}, lr={args.lr}",
        
        # 训练性能
        'training_time_seconds': training_summary['total_training_time_seconds'].iloc[0],
        'convergence_epoch': training_summary['convergence_epoch'].iloc[0],
        'best_val_q_median': training_summary['best_val_q_median'].iloc[0],
        'best_val_q_90': training_summary['best_val_q_90'].iloc[0], 
        'best_val_q_mean': training_summary['best_val_q_mean'].iloc[0],
        'best_val_correlation': training_summary['best_val_correlation'].iloc[0],
        
        # 工作负载性能
        'job_light_q_median': eval_score_jl['q_median'],
        'job_light_q_90': eval_score_jl['q_90'],
        'job_light_q_mean': eval_score_jl['q_mean'],
        'job_light_correlation': workload_results_jl['correlation'],
        'job_light_queries': workload_results_jl['total_queries'],
        
        'synthetic_q_median': eval_score_syn['q_median'],
        'synthetic_q_90': eval_score_syn['q_90'],
        'synthetic_q_mean': eval_score_syn['q_mean'],
        'synthetic_correlation': workload_results_syn['correlation'],
        'synthetic_queries': workload_results_syn['total_queries'],
        
        # 综合指标
        'avg_workload_q_median': (eval_score_jl['q_median'] + eval_score_syn['q_median']) / 2,
        'avg_workload_q_mean': (eval_score_jl['q_mean'] + eval_score_syn['q_mean']) / 2,
        'avg_workload_correlation': (workload_results_jl['correlation'] + workload_results_syn['correlation']) / 2,
        'total_test_queries': workload_results_jl['total_queries'] + workload_results_syn['total_queries']
    }
    
    # 保存最终报告
    final_report_df = pd.DataFrame([final_performance_report])
    final_report_df.to_csv(args.newpath + 'queryformer_final_performance_report.csv', index=False)
    
    print(f"All done. Results saved to {args.newpath}")
    
except Exception as e:
    print(f"Error creating final report: {e}")

# %%



