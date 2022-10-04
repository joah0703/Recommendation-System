# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 22:10:05 2022

@author: yurim
"""

import os
import pandas as pd
import datetime

os.chdir(r'E:\2022\0_study\2_추천시스템\220706')

raw_dt = pd.read_csv('../미디어패널 데이터/원시자료 데이터(세로통합)_20220225/d21v28_KMP_csv.csv')
raw_dt.dtypes
#(10154, 1150)


# =============================================================================
# 중복 제거 pid 추출
# 빈 DMM 생성
# =============================================================================
pid = list(set(raw_dt['pid']))
pid.sort()

col_names = ['pid']
col_names.extend(['P'+str(i) for i in range(1,18)])
col_names.extend(['M'+str(i) for i in range(1,43)])
col_names.extend(['A'+str(i) for i in range(1,45)])
col_names.extend(['C'+str(i) for i in range(1,22)])

dmm = pd.DataFrame(columns=col_names)
dmm['pid'] = pid
#(10154, 125)


#%%
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
#Run time : 0:49:52.170332


#%%

dmm.head(10)
dmm = dmm.fillna(0)

dmm.to_csv('0706_1_DMM.csv')
    






#%% 추가
# =============================================================================
# DMM을 통해 가장 많이 사용한 매체가 무엇인지 파악
# =============================================================================
# dmm = pd.read_csv(r'E:\2022\0_study\2_추천시스템\220313\0313_DMM.csv')
# dmm_sum = dmm.sum()

# M_sum = dmm_sum[2:44].sort_values(ascending=False)
# # M3     407356 가정용tv
# # M19    271369 스마트폰
# # M7     109837 데스크톱pc
# # M1      71301 신문/책/잡지
# # M8      32133 노트북pc
# # M24     21326 카오디오

# A_sum = dmm_sum[44:88].sort_values(ascending=False)
# # A1     292455 지상파tv 시청
# # A43     81712 음성통화
# # A24     77981 문서 작업 프로그램
# # A14     71191 책 읽기
# # A19     57720 채팅/메신저
# # A3      54456 비지상파tv 시청

# C_sum = dmm_sum[88:].sort_values(ascending=False)
# # C1     207524 케이블 tv 방송서비스를 통해
# # C2     179919 iptv 방송서비스를 통해
# # C11    124870 무선인터넷-wifi를 통해
# # C10    107917 이동통신 무선인터넷(3G 등) 통해
# # C9      83715 유선인터넷을 통해