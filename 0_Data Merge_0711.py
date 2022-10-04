# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 14:06:37 2022

@author: Yurim
"""

import os
import pandas as pd
import datetime
os.chdir(r'E:\2022\0_study\2_추천시스템\미디어패널 데이터\원시자료 데이터(세로통합)_20220225')

h_dt = pd.read_csv('h21v28_KMP_csv.csv', encoding='cp949')
d_dt = pd.read_csv('d21v28_KMP_csv.csv', encoding='cp949')
p_dt = pd.read_csv('p21v28_KMP_csv.csv', encoding='cp949')



##### 1) ‘hid’ 기준으로 D, H 데이터 left join
d_h = pd.merge(d_dt, h_dt, on='hid', how='left')


##### 2) D+H 데이터에서 ‘가족 구성(h21fly_typ)’이 3(부부+자녀), 4(조부모+부부+자녀)만 추출
d_h2 = d_h[(d_h['h21fly_typ']==3)|(d_h['h21fly_typ']==4)]
#(22719, 2028)


##### 3) 위에서 생성한 데이터로 LDA 데이터 생성
# =============================================================================
# 중복 제거 pid 추출
# 빈 DMM 생성
# =============================================================================
raw_dt = d_h2
pid = list(set(raw_dt['pid']))
pid.sort()

col_names = ['pid']
col_names.extend(['P'+str(i) for i in range(1,18)])
col_names.extend(['M'+str(i) for i in range(1,43)])
col_names.extend(['A'+str(i) for i in range(1,45)])
col_names.extend(['C'+str(i) for i in range(1,22)])

dmm = pd.DataFrame(columns=col_names)
dmm['pid'] = pid
#(7573, 125)


# =============================================================================
# DMM 값 채우기
# =============================================================================
start_time = datetime.datetime.now()

for pid_num in pid:
    
    # d21p
    subset = raw_dt[raw_dt['pid']==pid_num].loc[:,'d21p1':'d21p96']
    cnt = subset.apply(pd.value_counts).sum(axis=1)
    
    if 16 in cnt.index : cnt = cnt.drop(index=16) #기타
    if 9999 in cnt.index : cnt = cnt.drop(index=9999)

    for idx in cnt.index:
        dmm.loc[dmm.index[dmm['pid']==pid_num], 'P'+str(idx)] = int(cnt[idx])
        
        
    # d21MA, d21MB
    subset = pd.concat([raw_dt[raw_dt['pid']==pid_num].loc[:,'d21MA1':'d21MA96'], 
                       raw_dt[raw_dt['pid']==pid_num].loc[:,'d21MB1':'d21MB96']], axis=1)
    cnt = subset.apply(pd.value_counts).sum(axis=1)
    
    if 0 in cnt.index : cnt = cnt.drop(index=0)
    if 9999 in cnt.index : cnt = cnt.drop(index=9999)

    for idx in cnt.index:
        dmm.loc[dmm.index[dmm['pid']==pid_num], 'M'+str(idx)] = int(cnt[idx])
    
    
    # d21AA, d21AB
    subset = pd.concat([raw_dt[raw_dt['pid']==pid_num].loc[:,'d21AA1':'d21AA96'], 
                       raw_dt[raw_dt['pid']==pid_num].loc[:,'d21AB1':'d21AB96']], axis=1)
    cnt = subset.apply(pd.value_counts).sum(axis=1)
    
    if 0 in cnt.index : cnt = cnt.drop(index=0)
    if 9999 in cnt.index : cnt = cnt.drop(index=9999)

    for idx in cnt.index:
        dmm.loc[dmm.index[dmm['pid']==pid_num], 'A'+str(idx)] = int(cnt[idx])
    
    
    # d21CA, d21CB
    subset = pd.concat([raw_dt[raw_dt['pid']==pid_num].loc[:,'d21CA1':'d21CA96'], 
                       raw_dt[raw_dt['pid']==pid_num].loc[:,'d21CB1':'d21CB96']], axis=1)
    cnt = subset.apply(pd.value_counts).sum(axis=1)
    
    if 0 in cnt.index : cnt = cnt.drop(index=0)
    if 9999 in cnt.index : cnt = cnt.drop(index=9999)

    for idx in cnt.index:
        dmm.loc[dmm.index[dmm['pid']==pid_num], 'C'+str(idx)] = int(cnt[idx])
    
    print('pid', pid_num, round((pid.index(pid_num)+1)/len(pid)*100, 2), '% done!' )
    

end_time = datetime.datetime.now()
print('Run time :', end_time-start_time)
#Run time : 0:36:54.004385


#%%

dmm.head(10)
dmm = dmm.fillna(0)

dmm.to_csv(r'E:\2022\0_study\2_추천시스템\220711\0711_1_DMM.csv')
    