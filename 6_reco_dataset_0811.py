# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 09:56:34 2022

@author: Yurim
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import os
os.chdir(r'E:\2022\0_study\2_추천시스템\220811')

#%%
# =============================================================================
# create BPR input data
# =============================================================================
pid_topic = pd.read_csv('../220722/0711_3_pid_topic5.csv')
h_dt = pd.read_csv('../미디어패널 데이터/원시자료 데이터(세로통합)_20220225/h21v28_KMP_csv.csv', encoding='cp949')
p_dt = pd.read_csv('../미디어패널 데이터/원시자료 데이터(세로통합)_20220225/p21v28_KMP_csv.csv', encoding='cp949')

data2 = pd.merge(pid_topic, p_dt, on='pid', how='left')
data = pd.merge(h_dt, data2, on='hid', how='right')


col_name = ['pid','h21a01034','h21a01035','h21a01036','h21a01037','h21a03021',
            'h21a14027','h21a14029','h21a14031','h21a14033','h21a17041']
device = data[col_name]
device = pd.get_dummies(device, columns=['h21a14027','h21a14029','h21a14031','h21a14033','h21a17041'], 
                        prefix=['h21a14027','h21a14029','h21a14031','h21a14033','h21a17041'])
device[['h21a01034','h21a01035','h21a01036','h21a01037','h21a03021']] = device[['h21a01034','h21a01035','h21a01036','h21a01037','h21a03021']].replace(' ',0).apply(pd.to_numeric).replace(2,0)
device = device.astype('int')
device.drop(['h21a14027_ ','h21a14029_ ','h21a14031_ ','h21a14033_ ','h21a17041_ '], axis=1, inplace=True)


# h21a01034~h21a01037을 h21a01034로 통합
h21a010 = device[['h21a01034','h21a01035','h21a01036','h21a01037']].sum(axis=1)
h21a010_ = h21a010.apply(lambda x: 1 if x>0 else x)
device = device.drop(columns=['h21a01034','h21a01035','h21a01036','h21a01037'])
device['h21a01034'] = h21a010_


# h21a14027','h21a14029','h21a14031','h21a14033'을 h21a14027_...로 통합
a27 = [x[x.index('_')+1:] for x in device.columns.tolist() if "h21a14027" in x]
a29 = [x[x.index('_')+1:] for x in device.columns.tolist() if "h21a14029" in x]
a31 = [x[x.index('_')+1:] for x in device.columns.tolist() if "h21a14031" in x]
a33 = [x[x.index('_')+1:] for x in device.columns.tolist() if "h21a14033" in x]
col140 = {int(x):'h21a14027_'+x for x in set(a27+a29+a31+a33)}
col140_df = pd.DataFrame(np.zeros((len(device),len(col140))), columns=['h21a14027_'+x for x in set(a27+a29+a31+a33)])
col140_df['pid'] = device['pid']

sub = data[['pid','h21a14027','h21a14029','h21a14031','h21a14033']].replace(' ',0).apply(pd.to_numeric)
for i in tqdm(sub['pid']):
    check_value = sub[sub['pid']==i][['h21a14027','h21a14029','h21a14031','h21a14033']]
    if int(check_value.sum(axis=1))==0:
        continue

    for col in check_value.columns:
        if int(check_value[col])!=0:
            col140_df.loc[col140_df['pid']==i, col140[int(check_value[col])]] += 1
        
for i in ['h21a14027_'+x for x in set(a27+a29+a31+a33)]:
    col140_df[i] = col140_df[i].apply(lambda x: 1 if x>0 else 0)



device_sub = device[['pid','h21a01034','h21a03021']+[x for x in device.columns.tolist() if "h21a17041" in x]]
device = pd.merge(device_sub, col140_df, on='pid')
device = device.sort_index(axis=1)


#%%
### type 1
title_name = device.columns.tolist()
title_name.remove('pid')
devices = pd.DataFrame({'movie_id':[x for x in range(1,len(device.columns))], 
                        'title':title_name, 
                        'genres':[0 for i in range(1,len(device.columns))]})
devices.to_csv('bpr_devices.csv', header=False, index=False)


pid_id = []
movie_id = []
rating = []
col_name = title_name
i = 0
for pid in device['pid']:
    movie_pid = device.loc[device['pid']==pid,col_name].values[0]
    for idx,movie in enumerate(movie_pid):
        pid_id.append(pid)
        movie_id.append(idx+1)
        rating.append(movie)
    i+=1
    print(pid, "Done! : ", np.round(i/len(device['pid'])*100,4))
       
ratings = pd.DataFrame({'user_id':pid_id,
                        'movie_id':movie_id,
                        'rating':rating,
                        'timestamp':[0 for i in range(0,len(rating))]})

ratings.to_csv('bpr_ratings_01.csv', header=False, index=False)


#%%
### type 2: sum emotion score
pid_id = []
movie_id = []
rating = []
col_name = title_name
i = 0
for pid in device['pid']:
    movie_pid = device.loc[device['pid']==pid,col_name].values[0]
    for idx,movie in enumerate(movie_pid):
        pid_id.append(pid)
        movie_id.append(idx+1)
        if movie==1:
            rating.append(movie+3)
        else:
            rating.append(movie+1)
    i+=1
    print(pid, "Done! : ", np.round(i/len(device['pid'])*100,4))
        
ratings = pd.DataFrame({'user_id':pid_id,
                        'movie_id':movie_id,
                        'rating':rating,
                        'timestamp':[0 for i in range(0,len(rating))]})



### LDA
## read emotion score data
emotion_df = pd.read_csv('../220722/LDA data with emotion score.csv')
emotion_df = emotion_df[['pid','score_h','score_p']].fillna(0)
emotion_df['score'] = emotion_df[['score_h','score_p']].sum(axis=1)
emotion_df['score'] = emotion_df['score']/2

new_ratings = pd.merge(ratings, emotion_df, left_on='user_id',right_on='pid', how='left')
new_ratings['rating'] = new_ratings['rating']+new_ratings['score']
new_ratings.drop(['pid','score_h','score_p','score'], axis=1, inplace=True)

new_ratings.to_csv('bpr_ratings_LDAemotion.csv', header=False, index=False)



### KMeans
## read emotion score data
emotion_df = pd.read_csv('../220722/KMeans data with emotion score.csv')
emotion_df = emotion_df[['pid','score_h','score_p']].fillna(0)
emotion_df['score'] = emotion_df[['score_h','score_p']].sum(axis=1)
emotion_df['score'] = emotion_df['score']/2

new_ratings = pd.merge(ratings, emotion_df, left_on='user_id',right_on='pid', how='left')
new_ratings['rating'] = new_ratings['rating']+new_ratings['score']
new_ratings.drop(['pid','score_h','score_p','score'], axis=1, inplace=True)

new_ratings.to_csv('bpr_ratings_KMemotion.csv', header=False, index=False)

#점수가 0이 아닌 것만 저장
# new_ratings = new_ratings[new_ratings['rating']!=0] 
# new_ratings.to_csv('bpr_ratings2 without 0.csv', header=False, index=False)



