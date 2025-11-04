import pandas as pd
import numpy as np

# 加载数据时指定类型节省内存
dtypes = {
    'row_id': 'int64', 
    'timestamp': 'int64',
    'user_id': 'int32',
    'content_id': 'int16',
    'content_type_id': 'int8',
    'task_container_id': 'int16',
    'user_answer': 'int8',
    'answered_correctly': 'int8',
    'prior_question_elapsed_time': 'float32',
    'prior_question_had_explanation': 'boolean'
}
train = pd.read_csv('train.csv', dtype=dtypes)

# 过滤掉讲座事件（只保留问题）
train = train[train['content_type_id'] == 0].reset_index(drop=True)

# 合并题目元数据
questions = pd.read_csv('questions.csv')
train = train.merge(questions, left_on='content_id', right_on='question_id', how='left')

# 处理空值
train['prior_question_elapsed_time'] = train['prior_question_elapsed_time'].fillna(
    train['prior_question_elapsed_time'].median())
train['prior_question_had_explanation'] = train['prior_question_had_explanation'].fillna(False)

# 标签多热编码
tags_split = train['tags'].str.split(' ', expand=True)
tags_dummies = pd.get_dummies(tags_split.stack()).groupby(level=0).max()
train = pd.concat([train, tags_dummies], axis=1)

# 用户累计正确率
user_correct = train.groupby('user_id')['answered_correctly'].agg(['mean', 'count'])
user_correct.columns = ['user_hist_correct_rate', 'user_hist_question_count']
train = train.merge(user_correct, on='user_id', how='left')

# 用户最近20题正确率（滑动窗口）
train['user_recent_20_correct'] = train.groupby('user_id')['answered_correctly'].transform(
    lambda x: x.rolling(20, min_periods=1).mean()
)

# 题目全局正确率
question_diff = train.groupby('question_id')['answered_correctly'].mean().reset_index()
question_diff.columns = ['question_id', 'question_difficulty']
train = train.merge(question_diff, on='question_id', how='left')

# 题目在用户所属分组的难度（如TOEIC part）
part_diff = train.groupby('part')['answered_correctly'].mean().reset_index()
part_diff.columns = ['part', 'part_avg_correct']
train = train.merge(part_diff, on='part', how='left')

# 用户答题间隔时间变化率
train['time_diff_rate'] = train.groupby('user_id')['timestamp'].diff().fillna(0) / 1e3  # 转换为秒

# 用户当前任务容器与上次的时间差
train['task_container_gap'] = train.groupby('user_id')['task_container_id'].diff().fillna(0)

import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

# 特征列选择
features = [
    'user_hist_correct_rate', 'user_hist_question_count', 'user_recent_20_correct',
    'question_difficulty', 'part_avg_correct', 'prior_question_elapsed_time',
    'prior_question_had_explanation', 'time_diff_rate', 'task_container_gap'
] + list(tags_dummies.columns)

# 时间序列交叉验证
tscv = TimeSeriesSplit(n_splits=5)
X = train[features]
y = train['answered_correctly']

for fold, (train_idx, val_idx) in enumerate(tscv.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # 数据集转换
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val)
    
    # 参数设置（优化NPU兼容性）
    # params = {
    #     'objective': 'binary',
    #     'metric': 'auc',
    #     'device': 'cpu',  # 若使用英特尔NPU，需安装oneAPI优化版本
    #     'num_leaves': 31,
    #     'learning_rate': 0.05,
    #     'feature_fraction': 0.8
    # }

    # 定义 LightGBM 模型的参数，启用 GPU
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'device': 'gpu',  # 启用 GPU
        'gpu_device_id': 0,  # 0 为 GTX 1050 的设备 ID，通常是 0
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8
    }
    
    # 训练
    model = lgb.train(params, dtrain, valid_sets=[dval], 
                      callbacks=[lgb.log_evaluation(100)])
    
    # 保存模型
    model.save_model(f'lgbm_fold{fold}.txt')