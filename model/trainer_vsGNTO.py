import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .dataset import PlanTreeDataset
from .database_util import collator, get_job_table_sample
import os
import time
import torch
from scipy.stats import pearsonr

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def print_qerror(preds_unnorm, labels_unnorm, prints=True):
    """
    计算Q-error指标: 50, 75, 90, 95, 99 分位数
    Q-error = max(predicted/actual, actual/predicted)
    """
    qerror = []
    
    preds_unnorm = np.array(preds_unnorm)
    labels_unnorm = np.array(labels_unnorm)
    
    # 避免除零错误
    labels_unnorm = np.maximum(labels_unnorm, 1e-10)
    preds_unnorm = np.maximum(preds_unnorm, 1e-10)
    
    for i in range(len(preds_unnorm)):
        pred = float(preds_unnorm[i])
        actual = float(labels_unnorm[i])
        q_err = max(pred / actual, actual / pred)
        qerror.append(q_err)

    qerror = np.array(qerror)
    
    # 计算要求的5个分位数
    q_50 = np.median(qerror)
    q_75 = np.percentile(qerror, 75)
    q_90 = np.percentile(qerror, 90)
    q_95 = np.percentile(qerror, 95)
    q_99 = np.percentile(qerror, 99)
    q_mean = np.mean(qerror)

    if prints:
        print(f"Q-error: 50%={q_50:.4f}, 75%={q_75:.4f}, 90%={q_90:.4f}, 95%={q_95:.4f}, 99%={q_99:.4f}, Mean={q_mean:.4f}")

    res = {
        'q_50': q_50,
        'q_75': q_75,
        'q_90': q_90,
        'q_95': q_95,
        'q_99': q_99,
        'q_mean': q_mean,
    }

    return res

def get_corr(ps, ls): # unnormalised
    ps = np.array(ps)
    ls = np.array(ls)
    corr, _ = pearsonr(np.log(ps), np.log(ls))
    return corr


def eval_workload(workload, methods, save_results=False, save_path=None):
    pass

def evaluate(model, ds, bs, norm, device, prints=True, save_results=False, save_path=None, dataset_name="validation"):
    """
    评估模型性能并收集详细数据 (真实值, 预测值, 整体Q值分布)
    """
    import time
    
    model.eval()
    cost_predss = np.empty(0)

    with torch.no_grad():
        for i in range(0, len(ds), bs):
            batch, batch_labels = collator(list(zip(*[ds[j] for j in range(i,min(i+bs, len(ds)) ) ])))
            batch = batch.to(device)
            cost_preds, _ = model(batch)
            cost_preds = cost_preds.squeeze()
            cost_predss = np.append(cost_predss, cost_preds.cpu().detach().numpy())
    
    # 计算指标
    unnorm_preds = norm.unnormalize_labels(cost_predss)
    unnorm_labels = ds.costs
    
    # 收集Q-error统计 (50, 75, 90, 95, 99)
    scores = print_qerror(unnorm_preds, unnorm_labels, prints)
    
    # 保存详细结果 (真实值, 预测值)
    eval_results = {
        'dataset_name': dataset_name,
        'total_queries': len(ds),
        **scores
    }
    
    if save_results and save_path:
        # 保存预测结果详情: query_id, predicted_cost, actual_cost
        detailed_results = pd.DataFrame({
            'query_id': range(len(unnorm_preds)),
            'predicted_cost': unnorm_preds,
            'actual_cost': unnorm_labels,
            'q_error': [max(p/a, a/p) for p, a in zip(unnorm_preds, unnorm_labels)]
        })
        detailed_results.to_csv(f"{save_path}{dataset_name}_detailed_results.csv", index=False)
        
        # 保存评估总结
        pd.DataFrame([eval_results]).to_csv(f"{save_path}{dataset_name}_summary.csv", index=False)
        
        if prints:
            print(f"Results saved: {save_path}{dataset_name}_detailed_results.csv")
    
    return scores, 0, eval_results

def train(model, train_ds, val_ds, crit, cost_norm, args, optimizer=None, scheduler=None):
    
    to_pred, bs, device, epochs, clip_size = \
        args.to_predict, args.bs, args.device, args.epochs, args.clip_size
    lr = args.lr

    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if not scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.7)

    # 简化训练数据收集: 仅 Epoch, Loss, Time, Q-metrics
    training_log = [] 

    t0 = time.time()
    rng = np.random.default_rng()
    best_prev = 999999

    for epoch in range(epochs):
        epoch_start_time = time.time()
        losses = 0
        cost_predss = np.empty(0)
        total_grad_norm = 0
        num_batches = 0

        model.train()
        train_idxs = rng.permutation(len(train_ds))
        cost_labelss = np.array(train_ds.costs)[train_idxs]

        for idxs in chunks(train_idxs, bs):
            optimizer.zero_grad()
            batch, batch_labels = collator(list(zip(*[train_ds[j] for j in idxs])))
            l, r = zip(*(batch_labels))
            batch_cost_label = torch.FloatTensor(l).to(device)
            batch = batch.to(device)

            cost_preds, _ = model(batch)
            cost_preds = cost_preds.squeeze()

            loss = crit(cost_preds, batch_cost_label)
            loss.backward()

            # 计算梯度范数
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_size)
            total_grad_norm += grad_norm.item()
            num_batches += 1

            optimizer.step()
            
            del batch, batch_labels
            torch.cuda.empty_cache()

            losses += loss.item()
            # 仅在需要计算Q-error的epoch收集预测值以节省内存
            if epoch % 10 == 0 or epoch == epochs - 1:
                cost_predss = np.append(cost_predss, cost_preds.detach().cpu().numpy())

        avg_train_loss = losses / len(train_ds)
        avg_grad_norm = total_grad_norm / num_batches if num_batches > 0 else 0
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']

        # 每10个epoch进行一次详细记录
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f'Epoch: {epoch}, Loss: {avg_train_loss:.4f}, Time: {epoch_time:.2f}s, LR: {current_lr:.6f}, GradNorm: {avg_grad_norm:.4f}')
            
            # 计算训练集 Q-metrics
            train_scores = print_qerror(cost_norm.unnormalize_labels(cost_predss), cost_labelss, prints=True)
            
            log_entry = {
                'epoch': epoch,
                'loss': avg_train_loss,
                'time': epoch_time,
                'lr': current_lr,
                'grad_norm': avg_grad_norm,
                'train_q_50': train_scores['q_50'],
                'train_q_75': train_scores['q_75'],
                'train_q_90': train_scores['q_90'],
                'train_q_95': train_scores['q_95'],
                'train_q_99': train_scores['q_99']
            }
            
            # 验证集评估 (也是每10 epoch)
            # 不再为每个epoch保存单独的文件，而是记录到总日志中
            val_scores, _, _ = evaluate(
                model, val_ds, bs, cost_norm, device, prints=True,
                save_results=False, 
                save_path=getattr(args, 'newpath', './results/'),
                dataset_name=f"val_epoch_{epoch}"
            )
            
            # 记录验证集指标
            log_entry.update({
                'val_q_50': val_scores['q_50'],
                'val_q_75': val_scores['q_75'],
                'val_q_90': val_scores['q_90'],
                'val_q_95': val_scores['q_95'],
                'val_q_99': val_scores['q_99']
            })

            if val_scores['q_50'] < best_prev: # 使用Median作为最佳模型标准? 原代码是mean, 这里可以用Median或Mean
                best_prev = val_scores['q_50']
                if hasattr(args, 'newpath'):
                    torch.save(model.state_dict(), args.newpath + 'best_model.pt')

            training_log.append(log_entry)
            
            # 实时保存训练日志
            save_path = getattr(args, 'newpath', './results/')
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            pd.DataFrame(training_log).to_csv(save_path + 'training_log.csv', index=False)

        scheduler.step()

    return model, best_prev, pd.DataFrame(training_log)


def logging(args, epoch, qscores, filename = None, save_model = False, model = None):
    arg_keys = [attr for attr in dir(args) if not attr.startswith('__')]
    arg_vals = [getattr(args, attr) for attr in arg_keys]
    
    res = dict(zip(arg_keys, arg_vals))
    model_checkpoint = str(hash(tuple(arg_vals))) + '.pt'

    res['epoch'] = epoch
    res['model'] = model_checkpoint 


    res = {**res, **qscores}

    filename = args.newpath + filename
    model_checkpoint = args.newpath + model_checkpoint
    
    if filename is not None:
        if os.path.isfile(filename):
            df = pd.read_csv(filename)
            res_df = pd.DataFrame([res])
            df = pd.concat([df, res_df], ignore_index=True)
            df.to_csv(filename, index=False)
        else:
            df = pd.DataFrame(res, index=[0])
            df.to_csv(filename, index=False)
    if save_model:
        torch.save({
            'model': model.state_dict(),
            'args' : args
        }, model_checkpoint)
    
    return res['model']


def comprehensive_evaluation(model, datasets_dict, methods, save_path=None):
    """
    Comprehensive evaluation across multiple datasets and collect all data.
    Comprehensive evaluation across multiple datasets and collect all data
    
    Args:
        model: The trained model.
        datasets_dict: Dictionary of datasets in the format {'dataset_name': dataset_object}.
        methods: Dictionary of evaluation method configurations.
        save_path: Path to save the evaluation results.
    
    Returns:
        comprehensive_results: Overall comprehensive evaluation results.
    """
    import time
    
    print("=" * 60)
    print("Starting Comprehensive Evaluation")
    print("=" * 60)
    
    total_start_time = time.time()
    all_results = []
    
    # 评估每个数据集
    for dataset_name, dataset in datasets_dict.items():
        print(f"\n--- 评估数据集: {dataset_name} ---")
        
        if 'workload' in dataset_name.lower():
            # 如果是工作负载，使用eval_workload
            workload_name = dataset_name.replace('workload_', '')
            eval_score, ds, workload_results = eval_workload(
                workload_name, methods, save_results=True, save_path=save_path
            )
            all_results.append(workload_results)
        else:
            # 普通数据集评估
            eval_score, corr, eval_results = evaluate(
                model, dataset, methods['bs'], methods['cost_norm'], 
                methods['device'], prints=True, save_results=True,
                save_path=save_path, dataset_name=dataset_name
            )
            all_results.append(eval_results)
    
    total_eval_time = time.time() - total_start_time
    
    # 计算综合统计
    comprehensive_results = {
        'total_evaluation_time': total_eval_time,
        'num_datasets': len(datasets_dict),
        'total_queries': sum([r.get('total_queries', 0) for r in all_results]),
        'avg_q_median': np.mean([r.get('q_median', 0) for r in all_results]),
        'avg_q_90': np.mean([r.get('q_90', 0) for r in all_results]),
        'avg_q_mean': np.mean([r.get('q_mean', 0) for r in all_results]),
        'avg_correlation': np.mean([r.get('correlation', 0) for r in all_results]),
        'best_q_median': min([r.get('q_median', float('inf')) for r in all_results]),
        'worst_q_median': max([r.get('q_median', 0) for r in all_results]),
        'best_correlation': max([r.get('correlation', 0) for r in all_results]),
        'worst_correlation': min([r.get('correlation', 1) for r in all_results])
    }

    print(f"\n=== Comprehensive Results ===")
    print(f"Total queries: {comprehensive_results['total_queries']}")
    print(f"Average Q-error median: {comprehensive_results['avg_q_median']:.4f}")
    print(f"Average Q-error 90th: {comprehensive_results['avg_q_90']:.4f}")
    print(f"Average Q-error mean: {comprehensive_results['avg_q_mean']:.4f}")
    print(f"Average correlation: {comprehensive_results['avg_correlation']:.4f}")
    print(f"Best Q-error median: {comprehensive_results['best_q_median']:.4f}")
    print(f"Worst Q-error median: {comprehensive_results['worst_q_median']:.4f}")
    print(f"Total evaluation time: {total_eval_time:.2f} seconds")
    
    # 保存综合结果
    if save_path:
        # 保存详细结果
        detailed_df = pd.DataFrame(all_results)
        detailed_df.to_csv(f"{save_path}comprehensive_evaluation_detailed.csv", index=False)
        
        # 保存综合总结
        summary_df = pd.DataFrame([comprehensive_results])
        summary_df.to_csv(f"{save_path}comprehensive_evaluation_summary.csv", index=False)
        
        print(f"\nComprehensive evaluation results saved to:")
        print(f"  Detailed results: {save_path}comprehensive_evaluation_detailed.csv")
        print(f"  Comprehensive summary: {save_path}comprehensive_evaluation_summary.csv")
    
    return comprehensive_results, all_results  