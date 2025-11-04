import pandas as pd
import numpy as np
import gc
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from scipy.sparse import csr_matrix

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

# 1. 分块读取数据，按需过滤
chunk_size = 100000  # 根据内存情况调整
chunks = pd.read_csv('train.csv', dtype=dtypes, chunksize=chunk_size)
filtered_chunks = []

for chunk in chunks:
    # 过滤掉讲座事件（只保留问题）
    chunk = chunk[chunk['content_type_id'] == 0]
    chunk.drop(columns=['content_type_id'], inplace=True)  # 删除不需要的列
    filtered_chunks.append(chunk)

train = pd.concat(filtered_chunks, ignore_index=True)

# 2. 合并题目元数据
questions = pd.read_csv('questions.csv')
train = train.merge(questions, left_on='content_id', right_on='question_id', how='left')

# 3. 处理空值
train['prior_question_elapsed_time'] = train['prior_question_elapsed_time'].fillna(
    train['prior_question_elapsed_time'].median())
train['prior_question_had_explanation'] = train['prior_question_had_explanation'].fillna(False)

# 4. 标签多热编码
tags_series = train['tags'].astype(str).str.split()
unique_tags = set(tag for sublist in tags_series for tag in sublist)

# 使用稀疏矩阵处理标签
row_indices = []
col_indices = []
for i, tags in enumerate(tags_series):
    for tag in tags:
        col_idx = list(unique_tags).index(tag)
        row_indices.append(i)
        col_indices.append(col_idx)

tags_sparse = csr_matrix(([1] * len(row_indices), (row_indices, col_indices)))

# 将稀疏矩阵转换为DataFrame（可选）
tags_df = pd.DataFrame.sparse.from_spmatrix(tags_sparse, columns=unique_tags)
train = pd.concat([train, tags_df], axis=1)

# 5. 用户累计正确率
user_correct = train.groupby('user_id')['answered_correctly'].agg(['mean', 'count'])
user_correct.columns = ['user_hist_correct_rate', 'user_hist_question_count']
train = train.merge(user_correct, on='user_id', how='left')

# 6. 用户最近20题正确率（滑动窗口）
train['user_recent_20_correct'] = train.groupby('user_id')['answered_correctly'].transform(
    lambda x: x.rolling(20, min_periods=1).mean()
)

# 7. 题目全局正确率
question_diff = train.groupby('question_id')['answered_correctly'].mean().reset_index()
question_diff.columns = ['question_id', 'question_difficulty']
train = train.merge(question_diff, on='question_id', how='left')

# 8. 题目在用户所属分组的难度（如TOEIC part）
part_diff = train.groupby('part')['answered_correctly'].mean().reset_index()
part_diff.columns = ['part', 'part_avg_correct']
train = train.merge(part_diff, on='part', how='left')

# 9. 用户答题间隔时间变化率
train['time_diff_rate'] = train.groupby('user_id')['timestamp'].diff().fillna(0) / 1e3  # 转换为秒

# 10. 用户当前任务容器与上次的时间差
train['task_container_gap'] = train.groupby('user_id')['task_container_id'].diff().fillna(0)

# 释放不再需要的中间变量以节省内存
del questions, tags_series, tags_sparse, tags_df
gc.collect()

# 11. 特征列选择
features = [
    'user_hist_correct_rate', 'user_hist_question_count', 'user_recent_20_correct',
    'question_difficulty', 'part_avg_correct', 'prior_question_elapsed_time',
    'prior_question_had_explanation', 'time_diff_rate', 'task_container_gap'
] + list(tags_df.columns)

# 12. 时间序列交叉验证
tscv = TimeSeriesSplit(n_splits=5)
X = train[features]
y = train['answered_correctly']

# 13. 定义LightGBM模型训练
for fold, (train_idx, val_idx) in enumerate(tscv.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # 数据集转换
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val)
    
    # 训练参数配置
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'device': 'cpu',  # 改为使用CPU，防止GPU显存不足
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8
    }
    
    # 训练模型
    model = lgb.train(params, dtrain, valid_sets=[dval], 
                      callbacks=[lgb.log_evaluation(100)])
    
    # 保存模型
    model.save_model(f'lgbm_fold{fold}.txt')
    
    # 释放内存
    del dtrain, dval, model
    gc.collect()

# 最终内存优化
float_cols = train.select_dtypes(include=['float64']).columns
train[float_cols] = train[float_cols].astype('float32')

int_cols = train.select_dtypes(include=['int64']).columns
train[int_cols] = train[int_cols].astype('int32')

# 删除标签列，减少内存占用
train.drop(columns=['tags'], inplace=True)
gc.collect()
