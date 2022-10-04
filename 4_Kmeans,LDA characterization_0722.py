# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 15:44:56 2022

@author: Yurim
"""

import os
os.chdir(r'E:\2022\0_study\2_추천시스템\220722')
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Dunn_utils import get_Dunn_index
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.metrics import silhouette_score, silhouette_samples

pid_topic = pd.read_csv('0722_kmeans5.csv', index_col=0)
pid_topic_lda = pd.read_csv('0711_3_pid_topic5.csv')
pid_topic_lda = pid_topic_lda[['pid','가장 비중이 높은 토픽']]

D_dt = pd.read_csv('../미디어패널 데이터/원시자료 데이터(세로통합)_20220225/p21v28_KMP_csv.csv', encoding='cp949')
with open('0722_kmeans_input.pkl', 'rb') as f:
    kdata = pickle.load(f)


vari_col = ['pid', 'p21area', 'p21hhldsiz', 'p21houstyp', 'p21gender', 'p21age', 'p21age1', 
            'p21school', 'p21mar', 'p21income1', 'p21income', 'p21job1', 'p21job2']
person_dt = D_dt[vari_col]

merge_dt = pd.merge(pid_topic, person_dt)
merge_dt.rename(columns={'k_group':'topic'}, inplace=True)

merge_dt_lda = pd.merge(pid_topic_lda, person_dt)
merge_dt_lda.rename(columns={'가장 비중이 높은 토픽':'topic'}, inplace=True)



#%%
# =============================================================================
# Dunn index & 실루엣 계수
# =============================================================================
## Dunn Index
X = np.array(kdata)
labels = merge_dt['topic']
labels_lda = merge_dt_lda['topic']

intra_cluster_distance_type = 'cmpl_dd'
inter_cluster_distance_type = 'av_cent_ld'

print('Dunn index of KMeans :', get_Dunn_index(X, labels, intra_cluster_distance_type, inter_cluster_distance_type))
print('Dunn index of LDA :', get_Dunn_index(X, labels_lda, intra_cluster_distance_type, inter_cluster_distance_type))


## Silhouette
#----- 개별(클러스터 별) 실루엣 계수 계산
sil_score = silhouette_samples(X, labels)
cluster_sil = pd.DataFrame({'silouette_score':sil_score, 'cluster':labels})
print('Silhouette of KMeans')
cluster_sil.groupby('cluster')['silouette_score'].mean()

sil_score = silhouette_samples(X, labels_lda)
cluster_sil = pd.DataFrame({'silouette_score':sil_score, 'cluster':labels_lda})
print('Silhouette of LDA')
cluster_sil.groupby('cluster')['silouette_score'].mean()


#----- 전체 실루엣 계수 계산
print('Silhouette of KMeans :', silhouette_score(X, labels, metric="euclidean"))
print('Silhouette of LDA :', silhouette_score(X, labels_lda, metric="euclidean"))



#%%



#%%
# =============================================================================
# 토픽 별 개인 특성 파악 _ count ver
# =============================================================================
topic_num = 0

view_dt = merge_dt[merge_dt['topic']==topic_num]
all_col = ['p21area', 'p21hhldsiz', 'p21houstyp', 'p21gender', 'p21age', 'p21age1', 
            'p21school', 'p21mar', 'p21income1', 'p21income', 'p21job1', 'p21job2']
for col in all_col:
    print(col)
    print(view_dt[col].value_counts())



#%%
# =============================================================================
# 토픽 별 개인 특성 파악 _ plot ver
# =============================================================================
# topic_num = 0
# data_look = merge_dt
# data_topic = data_look[data_look['topic']==topic_num]


# all_col = ['p21area', 'p21hhldsiz', 'p21houstyp', 'p21gender', 'p21age', 'p21age1', 
#             'p21school', 'p21mar', 'p21income1', 'p21income', 'p21job1', 'p21job2']
# for col in all_col:
#     cnt_table = data_topic[col].value_counts()
    
#     plt.bar(list(cnt_table.keys()), list(cnt_table.values))
#     plt.title('Topic'+str(topic_num)+'_'+col)
#     plt.show()