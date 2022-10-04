# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 09:56:34 2022

@author: Yurim
"""

# =============================================================================
# BPR
# https://gitee.com/fruitwater/recommenders/blob/master/examples/02_model_collaborative_filtering/cornac_bpr_deep_dive.ipynb
# https://github.com/microsoft/recommenders/blob/main/examples/02_model_collaborative_filtering/cornac_bpr_deep_dive.ipynb
# =============================================================================
import sys
sys.path.append(r"E:\2022\0_study\2_추천시스템\recommendation code\recommenders-main")
import os
import cornac
#import papermill as pm
import pandas as pd
import random
from recommenders.datasets.python_splitters import python_random_split
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from recommenders.models.cornac.cornac_utils import predict_ranking
from recommenders.utils.timer import Timer
from recommenders.utils.constants import SEED
os.chdir(r'E:\2022\0_study\2_추천시스템\220811')


# Select MovieLens data size: 100k, 1m, 10m, or 20m
# MOVIELENS_DATA_SIZE = '100k'




# data = movielens.load_pandas_df(
#     size=MOVIELENS_DATA_SIZE,
#     header=["userID", "itemID", "rating"]
# )

# data.head()

raw_data = pd.read_csv('bpr_ratings_01.csv', header=None)
raw_data.columns = ['userID','itemID','raw_rating','timestemp']

####### 데이터셋 3개 중 하나 선택
# data = pd.read_csv('bpr_ratings_01.csv', header=None)
data = pd.read_csv('bpr_ratings_KMemotion.csv', header=None)
data = pd.read_csv('bpr_ratings_LDAemotion.csv', header=None)

# data = pd.read_csv('../220817/bpr_ratings_KMemotion.csv', header=None)
# data = pd.read_csv('../220817/bpr_ratings_LDAemotion.csv', header=None)


data.columns = ['userID','itemID','rating','timestemp']
del data['timestemp']
data['raw_rating'] = raw_data['raw_rating']

data = data[data['raw_rating']!=0]
data = data.reset_index(drop=True)


# print(data['rating'].min(), data['rating'].max())
# data['rating'] = (data['rating']*10).astype('int')
# data.groupby('itemID').apply(lambda x: pd.Series(x['rating'].values, index=x['userID'])).unstack()


####### train test split
# random.seed(1)
# test_size = 0.2
# ts_user = random.sample(data['userID'].unique().tolist(), int(len(data['userID'].unique())*test_size))
# tr_user = [x for x in data['userID'].unique() if x not in ts_user]

# test = pd.merge(data, pd.DataFrame({'userID':ts_user}), on='userID', how='right')
# train = pd.merge(data, pd.DataFrame({'userID':tr_user}), on='userID', how='right')


train, test = python_random_split(data, 0.9) #default = seed=42

del train['raw_rating']
# del test['raw_rating']
train_set = cornac.data.Dataset.from_uir(train.itertuples(index=False), seed=SEED)
print('Number of users: {}'.format(train_set.num_users))
print('Number of items: {}'.format(train_set.num_items))


# Model parameters
NUM_FACTORS = train_set.num_items #10 #200
NUM_EPOCHS = 100


bpr = cornac.models.BPR(
    k=NUM_FACTORS,
    max_iter=NUM_EPOCHS,
    learning_rate=0.01,
    lambda_reg=0.001,
    verbose=True,
    seed=SEED
)

with Timer() as t:
    bpr.fit(train_set)
print("Took {} seconds for training.".format(t))


with Timer() as t:
    all_predictions = predict_ranking(bpr, train, usercol='userID', itemcol='itemID', remove_seen=True)
print("Took {} seconds for prediction.".format(t))
print(all_predictions.head())



### metrics format으로 변경
acc_test = test[test['raw_rating']!=0]
del acc_test['raw_rating']


# top k items to recommend
TOP_K = 2
eval_map = map_at_k(acc_test, all_predictions, col_prediction='prediction', k=TOP_K)
eval_ndcg = ndcg_at_k(acc_test, all_predictions, col_prediction='prediction', k=TOP_K)
eval_precision = precision_at_k(acc_test, all_predictions, col_prediction='prediction', k=TOP_K)
eval_recall = recall_at_k(acc_test, all_predictions, col_prediction='prediction', k=TOP_K)
# 이진분류에서만 사용
#eval_auc = auc(test, all_predictions, col_prediction='prediction')

print("MAP:\t%f" % eval_map,
      "NDCG:\t%f" % eval_ndcg,
      "Precision@K:\t%f" % eval_precision,
      "Recall@K:\t%f" % eval_recall, sep='\n')
      #"AUC:\t%f" % eval_auc, sep='\n') #이진분류에서만 사용



#%%

# total_result = pd.merge(all_predictions, test, on=['userID','itemID'], how='right')
# total_result = total_result.sort_values(by=["userID", "prediction"], ascending=[True,False]) 

# total_result.to_csv('../220817/BPR_result_01.csv')


########## 그룹별 추천 결과 저장
pred_sort = all_predictions.sort_values(by=['userID','prediction'], ascending=[True, False])
pred_sort = pred_sort.reset_index()
del pred_sort['index']
#pred_sort.to_csv('../220817/BPR_prediction.csv')

del_idx = []
for user in pred_sort['userID'].unique().tolist():
    sub = pred_sort[pred_sort['userID']==user]
    del_idx.extend(sub[2:].index)

total_pred = pred_sort.drop(index=del_idx)


## test인 사람들에게 2개씩 추천할 경우
result = pd.merge(total_pred, test, on='userID', how='inner')
result  = result.drop(['itemID_y','rating','raw_rating'], axis=1)
result.rename(columns={'itemID_x':'itemID'}, inplace=True)
result = pd.merge(result, raw_data, on=['userID','itemID'], how='left')

group = pd.read_csv('../220722/0711_3_pid_topic5.csv')
group = group[['pid','가장 비중이 높은 토픽']]
group.rename(columns={'가장 비중이 높은 토픽':'topic'}, inplace=True)

result = pd.merge(result, group, left_on='userID', right_on='pid', how='left')
result.to_csv('../220817/BPR_prediction_2item.csv')


## 결과 종합
# 기존 사용자가 사용하던 아이템
test_topic = pd.merge(test, group, left_on='userID', right_on='pid', how='left')
test_topic.to_csv('../220817/test_topic.csv')
test_pivot = pd.pivot_table(test_topic, index='topic', columns='itemID', values='itemID', aggfunc='count')


#%%

# 토닥토닥 파이썬-추천 시스템을 위한 머신러닝 (위키독스)

# =============================================================================
# Item based 
# =============================================================================

import numpy as np
from math import sqrt
from tqdm import tqdm_notebook as tqdm


# 각자 작업 환경에 맞는 경로를 지정해주세요. Google Colab과 Jupyter환경에서 경로가 다를 수 있습니다.
#path = '/content/drive/MyDrive/data/movielens'
#ratings_df = pd.read_csv(os.path.join(path, 'ratings.csv'), encoding='utf-8')

# raw_data = pd.read_csv('bpr_ratings_01.csv', names=['userID','itemID','raw_rating','timestemp'])

####### 데이터셋 3개 중 하나 선택
# ratings_df = pd.read_csv('bpr_ratings_01.csv', names=['userID', 'itemID', 'rating', 'timestamp'])
# ratings_df = pd.read_csv('bpr_ratings_KMemotion.csv', names=['userID', 'itemID', 'rating', 'timestamp'])
# ratings_df = pd.read_csv('bpr_ratings_LDAemotion.csv', names=['userID', 'itemID', 'rating', 'timestamp'])
ratings_df = data
# ratings_df = pd.read_csv('../220817/bpr_ratings_KMemotion.csv', names=['userID', 'itemID', 'rating', 'timestamp'])
# ratings_df = pd.read_csv('../220817/bpr_ratings_LDAemotion.csv', names=['userID', 'itemID', 'rating', 'timestamp'])


# del ratings_df['timestamp']
# ratings_df['raw_rating'] = raw_data['raw_rating']

# print(ratings_df.shape)
# print(ratings_df.head())

# ratings_df = ratings_df[ratings_df['raw_rating']!=0]


####### train test split
# random.seed(1)
# test_size = 0.2
# ts_user = random.sample(ratings_df['userID'].unique().tolist(), int(len(ratings_df['userID'].unique())*test_size))
# tr_user = [x for x in ratings_df['userID'].unique() if x not in ts_user]

# test_df = pd.merge(ratings_df, pd.DataFrame({'userID':ts_user}), on='userID', how='right')
# train_df = pd.merge(ratings_df, pd.DataFrame({'userID':tr_user}), on='userID', how='right')


# train_df, test_df = train_test_split(ratings_df, test_size=0.9)#, stratify=ratings_df['raw_rating'], random_state=1)
train_df, test_df = train, test

# del train_df['raw_rating']
# del test_df['raw_rating']

# print(train_df.shape)
# print(test_df.shape)

sparse_matrix = train_df.groupby('itemID').apply(lambda x: pd.Series(x['rating'].values, index=x['userID'])).unstack()
# sparse_matrix = train_df.groupby('itemID').apply(lambda x: pd.Series(x['rating'].values, index=x['userID']))
sparse_matrix.index.name = 'itemID'

sparse_matrix

from sklearn.metrics.pairwise import cosine_similarity

def cossim_matrix(a, b):
    cossim_values = cosine_similarity(a.values, b.values)
    cossim_df = pd.DataFrame(data=cossim_values, columns = a.index.values, index=a.index)

    return cossim_df

item_sparse_matrix = sparse_matrix.fillna(0)
item_sparse_matrix.shape

item_cossim_df = cossim_matrix(item_sparse_matrix, item_sparse_matrix)
item_cossim_df

# movieId: 8938개, userId: 610개
# train_df에 포함된 userId를 계산에 반영한다
userId_grouped = train_df.groupby('userID')
# index: userId, columns: total movieId
item_prediction_result_df = pd.DataFrame(index=list(userId_grouped.indices.keys()), columns=item_sparse_matrix.index)
item_prediction_result_df

for userId, group in tqdm(userId_grouped):
    # user가 rating한 movieId * 전체 movieId
    user_sim = item_cossim_df.loc[group['itemID']]
    # user가 rating한 movieId * 1
    user_rating = group['rating']
    # 전체 movieId * 1
    sim_sum = user_sim.sum(axis=0)

    # userId의 전체 rating predictions (8938 * 1)
    pred_ratings = np.matmul(user_sim.T.to_numpy(), user_rating) / (sim_sum+1)
    item_prediction_result_df.loc[userId] = pred_ratings

item_prediction_result_df.head(10)



# =============================================================================
# 정확도 계산
# =============================================================================
item_prediction_new = item_prediction_result_df.stack()
item_prediction_new = pd.DataFrame(item_prediction_new).reset_index()
item_prediction_new.columns = ['userID', 'itemID', 'prediction']
item_prediction_new['prediction'] = item_prediction_new['prediction'].astype('float64')
item_prediction_new_input = item_prediction_new.loc[test_df.index,:]

acc_test = test_df[test_df['raw_rating']!=0]
del acc_test['raw_rating']


k = 2
eval_map = map_at_k(test_df, item_prediction_new_input, col_prediction='prediction', k=k)
eval_ndcg = ndcg_at_k(test_df, item_prediction_new_input, col_prediction='prediction', k=k)
eval_precision = precision_at_k(test_df, item_prediction_new_input, col_prediction='prediction', k=k)
eval_recall = recall_at_k(test_df, item_prediction_new_input, col_prediction='prediction', k=k)
# 이진분류에서만 사용
#eval_auc = auc(test, all_predictions, col_prediction='prediction')


print("MAP:\t%f" % eval_map,
      "NDCG:\t%f" % eval_ndcg,
      "Precision@K:\t%f" % eval_precision,
      "Recall@K:\t%f" % eval_recall, sep='\n')


#%%
###### binary precision
def precision_at_k(y_true, y_score, k, pos_label=1):
    from sklearn.utils import column_or_1d
    from sklearn.utils.multiclass import type_of_target
    
    y_true_type = type_of_target(y_true)
    if not (y_true_type == "binary"):
        raise ValueError("y_true must be a binary column.")
    
    # Makes this compatible with various array types
    y_true_arr = column_or_1d(y_true)
    y_score_arr = column_or_1d(y_score)
    
    y_true_arr = y_true_arr == pos_label
    
    desc_sort_order = np.argsort(y_score_arr)[::-1]
    y_true_sorted = y_true_arr[desc_sort_order]
    y_score_sorted = y_score_arr[desc_sort_order]
    
    true_positives = y_true_sorted[:k].sum()
    
    return true_positives / k

precision_at_k(test_df['rating'], item_prediction_new_input['prediction'], k=k)



#%%

total_result = pd.merge(item_prediction_new_input, test_df, on=['userID','itemID'], how='right')
total_result = total_result.sort_values(by=["userID", "prediction"], ascending=[True,False]) 
total_result.to_csv('IBCF_result_01.csv')




#%%
# =============================================================================
# User based
# =============================================================================
import os
import pandas as pd
import numpy as np
from math import sqrt
from tqdm import tqdm_notebook as tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 각자 작업 환경에 맞는 경로를 지정해주세요. Google Colab과 Jupyter환경에서 경로가 다를 수 있습니다.
#path = '/content/drive/MyDrive/data/movielens'
#ratings_df = pd.read_csv(os.path.join(path, 'ratings.csv'), encoding='utf-8')
raw_data = pd.read_csv('bpr_ratings_01.csv', names=['userID','itemID','raw_rating','timestemp'])

####### 데이터셋 3개 중 하나 선택
#ratings_df = pd.read_csv('bpr_ratings_01.csv', names=['userID', 'itemID', 'rating', 'timestamp'])
ratings_df = pd.read_csv('bpr_ratings_KMemotion.csv', names=['userID', 'itemID', 'rating', 'timestamp'])
ratings_df = pd.read_csv('bpr_ratings_LDAemotion.csv', names=['userID', 'itemID', 'rating', 'timestamp'])


del ratings_df['timestamp']
ratings_df['raw_rating'] = raw_data['raw_rating']

print(ratings_df.shape)
print(ratings_df.head())

ratings_df = ratings_df[ratings_df['raw_rating']!=0]
ratings_df = ratings_df.reset_index()


train_df, test_df = train_test_split(ratings_df, test_size=0.1)#, stratify=ratings_df['raw_rating'], random_state=1)
del train_df['raw_rating']
# del test_df['raw_rating']

print(train_df.shape)
print(test_df.shape)

sparse_matrix = train_df.groupby('itemID').apply(lambda x: pd.Series(x['rating'].values, index=x['userID'])).unstack()
sparse_matrix.index.name = 'itemID'

sparse_matrix

from sklearn.metrics.pairwise import cosine_similarity

def cossim_matrix(a, b):
    cossim_values = cosine_similarity(a.values, b.values)
    cossim_df = pd.DataFrame(data=cossim_values, columns = a.index.values, index=a.index)

    return cossim_df

user_sparse_matrix = sparse_matrix.fillna(0).transpose()

user_sparse_matrix.head(5)

user_sparse_matrix.shape

user_cossim_df = cossim_matrix(user_sparse_matrix, user_sparse_matrix)
user_cossim_df

movieId_grouped = train_df.groupby('itemID')
user_prediction_result_df = pd.DataFrame(index=list(movieId_grouped.indices.keys()), columns=user_sparse_matrix.index)
user_prediction_result_df

for movieId, group in tqdm(movieId_grouped):
    user_sim = user_cossim_df.loc[group['userID']]
    user_rating = group['rating']
    sim_sum = user_sim.sum(axis=0)

    pred_ratings = np.matmul(user_sim.T.to_numpy(), user_rating) / (sim_sum+1)
    user_prediction_result_df.loc[movieId] = pred_ratings

# return user_prediction_result_df.transpose()

##print(item_prediction_result_df.shape)
print(user_prediction_result_df.transpose().shape)

# 전체 user가 모든 movieId에 매긴 평점
##print(item_prediction_result_df.head())
print(user_prediction_result_df.transpose().head())

user_prediction_result_df = user_prediction_result_df.transpose()



# =============================================================================
# 정확도 계산
# =============================================================================
user_prediction_new = user_prediction_result_df.stack()
user_prediction_new = pd.DataFrame(user_prediction_new).reset_index()
user_prediction_new.columns = ['userID', 'itemID', 'prediction']
user_prediction_new['prediction'] = user_prediction_new['prediction'].astype('float64')
user_prediction_new_input = user_prediction_new.loc[test_df.index,:]

acc_test = test_df[test_df['raw_rating']!=0]
del acc_test['raw_rating']


k = 2
eval_map = map_at_k(acc_test, user_prediction_new_input, col_prediction='prediction', k=k)
eval_ndcg = ndcg_at_k(acc_test, user_prediction_new_input, col_prediction='prediction', k=k)
eval_precision = precision_at_k(acc_test, user_prediction_new_input, col_prediction='prediction', k=k)
eval_recall = recall_at_k(acc_test, user_prediction_new_input, col_prediction='prediction', k=k)
# 이진분류에서만 사용
#eval_auc = auc(test, all_predictions, col_prediction='prediction')


print("MAP:\t%f" % eval_map,
      "NDCG:\t%f" % eval_ndcg,
      "Precision@K:\t%f" % eval_precision,
      "Recall@K:\t%f" % eval_recall, sep='\n')


#%%


total_result = pd.merge(user_prediction_new_input, test_df, on=['userID','itemID'], how='right')
total_result = total_result.sort_values(by=["userID", "prediction"], ascending=[True,False]) 
total_result.to_csv('UBCF_result_LDA.csv')
