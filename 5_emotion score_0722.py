# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 19:27:45 2022

@author: Yurim
"""

import os
import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
os.chdir(r'E:\2022\0_study\2_추천시스템\220722')


pid_topic = pd.read_csv('0711_3_pid_topic5.csv')
pid_topic.rename(columns={'가장 비중이 높은 토픽':'topic'}, inplace=True)
pid_topic = pid_topic[['pid','topic']]
pid_topic_KM = pd.read_csv('0722_kmeans5.csv', index_col=0)
pid_topic_KM .rename(columns={'k_group':'topic'}, inplace=True)


h_dt = pd.read_csv('../미디어패널 데이터/원시자료 데이터(세로통합)_20220225/h21v28_KMP_csv.csv', encoding='cp949')
p_dt = pd.read_csv('../미디어패널 데이터/원시자료 데이터(세로통합)_20220225/p21v28_KMP_csv.csv', encoding='cp949')

p_topic = pd.merge(p_dt, pid_topic, on='pid', how='right')
total_df = pd.merge(p_topic, h_dt, on='hid', how='left')

p_topic_KM = pd.merge(p_dt, pid_topic_KM, on='pid', how='right')
total_df_KM = pd.merge(p_topic_KM, h_dt, on='hid', how='left')


#%%
### LDA 

# 1. 자녀의 TV, 인터넷, 스마트기기에 관한 염려
h_score_all = total_df[['pid','hid','topic','p21rel','h21d01023','h21d02023','h21d03024','h21d04020']].replace({' ':0, '1':1, '2':-1})

new_h_df = pd.DataFrame(columns=['topic','pos_num','neg_num'])
for i in range(0,len(h_score_all['topic'].unique())):
    ## 토픽 클러스터 별 계산
    h_score = h_score_all[h_score_all['topic']==i]
    # 긍부정 개수 세기
    pos_num = h_score[['h21d01023','h21d02023','h21d03024','h21d04020']].apply(lambda x: (x==1).sum())
    neg_num = h_score[['h21d01023','h21d02023','h21d03024','h21d04020']].apply(lambda x: (x==-1).sum())
    new_h_df = new_h_df.append({'topic':i, 'pos_num':pos_num.sum(), 'neg_num':neg_num.sum()}, ignore_index=True)

new_h_df['score'] = new_h_df.apply(lambda x: (x.pos_num-x.neg_num)/(x.pos_num+x.neg_num), axis=1)
total_df = pd.merge(total_df, new_h_df, on='topic', how='left')

# p21rel(가구주와의 관계)가 3(가구주의 자녀)인 가구원은 점수=0
total_df.loc[total_df['p21rel']==3,'score'] = 0



# 2. 프라이버시
p_score_all = total_df[['pid','hid','topic','p21d23001','p21d23002','p21d23003','p21d23004','p21d23005','p21d23006','p21d23007','p21d23008']]

new_p_df = pd.DataFrame(columns=['topic','pos_num','neg_num'])
for i in range(0,len(p_score_all['topic'].unique())):
    ## 토픽 클러스터 별 계산
    p_score = p_score_all[p_score_all['topic']==i]
    # 긍부정 개수 세기(1 or 2 / 4 or 5)
    pos_num = p_score[['p21d23001','p21d23002','p21d23003','p21d23004','p21d23005','p21d23006','p21d23007','p21d23008']].apply(lambda x: ((x==1)|(x==2)).sum())
    neg_num = p_score[['p21d23001','p21d23002','p21d23003','p21d23004','p21d23005','p21d23006','p21d23007','p21d23008']].apply(lambda x: ((x==4)|(x==5)).sum())
    new_p_df = new_p_df.append({'topic':i, 'pos_num':pos_num.sum(), 'neg_num':neg_num.sum()}, ignore_index=True)                                                                                                                     

new_p_df['score'] = new_p_df.apply(lambda x: (x.pos_num-x.neg_num)/(x.pos_num+x.neg_num), axis=1)
total_df = pd.merge(total_df, new_p_df, on='topic', how='left', suffixes=['_h','_p'])  


total_df.to_csv('LDA data with emotion score.csv')


#%%
### KMeans

# 1. 자녀의 TV, 인터넷, 스마트기기에 관한 염려
h_score_all = total_df_KM[['pid','hid','topic','p21rel','h21d01023','h21d02023','h21d03024','h21d04020']].replace({' ':0, '1':1, '2':-1})

new_h_df = pd.DataFrame(columns=['topic','pos_num','neg_num'])
for i in range(0,len(h_score_all['topic'].unique())):
    ## 토픽 클러스터 별 계산
    h_score = h_score_all[h_score_all['topic']==i]
    # 긍부정 개수 세기
    pos_num = h_score[['h21d01023','h21d02023','h21d03024','h21d04020']].apply(lambda x: (x==1).sum())
    neg_num = h_score[['h21d01023','h21d02023','h21d03024','h21d04020']].apply(lambda x: (x==-1).sum())
    new_h_df = new_h_df.append({'topic':i, 'pos_num':pos_num.sum(), 'neg_num':neg_num.sum()}, ignore_index=True)

new_h_df['score'] = new_h_df.apply(lambda x: (x.pos_num-x.neg_num)/(x.pos_num+x.neg_num), axis=1)
total_df_KM = pd.merge(total_df_KM, new_h_df, on='topic', how='left')

# p21rel(가구주와의 관계)가 3(가구주의 자녀)인 가구원은 점수=0
total_df_KM.loc[total_df_KM['p21rel']==3,'score'] = 0



# 2. 프라이버시
p_score_all = total_df_KM[['pid','hid','topic','p21d23001','p21d23002','p21d23003','p21d23004','p21d23005','p21d23006','p21d23007','p21d23008']]

new_p_df = pd.DataFrame(columns=['topic','pos_num','neg_num'])
for i in range(0,len(p_score_all['topic'].unique())):
    ## 토픽 클러스터 별 계산
    p_score = p_score_all[p_score_all['topic']==i]
    # 긍부정 개수 세기(1 or 2 / 4 or 5)
    pos_num = p_score[['p21d23001','p21d23002','p21d23003','p21d23004','p21d23005','p21d23006','p21d23007','p21d23008']].apply(lambda x: ((x==1)|(x==2)).sum())
    neg_num = p_score[['p21d23001','p21d23002','p21d23003','p21d23004','p21d23005','p21d23006','p21d23007','p21d23008']].apply(lambda x: ((x==4)|(x==5)).sum())
    new_p_df = new_p_df.append({'topic':i, 'pos_num':pos_num.sum(), 'neg_num':neg_num.sum()}, ignore_index=True)                                                                                                                     

new_p_df['score'] = new_p_df.apply(lambda x: (x.pos_num-x.neg_num)/(x.pos_num+x.neg_num), axis=1)
total_df_KM = pd.merge(total_df_KM, new_p_df, on='topic', how='left', suffixes=['_h','_p'])  


total_df_KM.to_csv('KMeans data with emotion score.csv')



