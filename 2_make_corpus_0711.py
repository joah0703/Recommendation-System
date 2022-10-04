# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 08:50:42 2022

@author: Yurim

make corpus and LDA
"""

import os
import pandas as pd
import numpy as np
import datetime
import itertools
import pickle

#import gensim
#from gensim import corpora

os.chdir(r'E:\2022\0_study\2_추천시스템\220711')

raw_dmm = pd.read_csv('0711_1_DMM.csv')
del raw_dmm['Unnamed: 0']
raw_dmm.dtypes
#raw_dt['d21MA1']


p_code = pd.read_csv('../미디어패널 데이터/place_code.csv')
m_code = pd.read_csv('../미디어패널 데이터/media_code.csv')
a_code = pd.read_csv('../미디어패널 데이터/action_code.csv')
c_code = pd.read_csv('../미디어패널 데이터/concat_code.csv')

p_code_dic = dict([("P"+str(i),a) for i, a in zip(p_code.code, p_code.place)])
del(p_code_dic['P16']) #기타
del(p_code_dic['P9999'])
m_code_dic = dict([("M"+str(i),a) for i, a in zip(m_code.code, m_code.media)])
del(m_code_dic['M0'])
del(m_code_dic['M9999'])
a_code_dic = dict([("A"+str(i),a) for i, a in zip(a_code.code, a_code.action)])
del(a_code_dic['A0'])
del(a_code_dic['A9999'])
c_code_dic = dict([("C"+str(i),a) for i, a in zip(c_code.code, c_code.concat)])
del(c_code_dic['C0'])
del(c_code_dic['C9999'])

code_dic = p_code_dic
code_dic.update(m_code_dic)
code_dic.update(a_code_dic)
code_dic.update(c_code_dic)
print(code_dic)



#%%
# =============================================================================
# corpus list 생성
# =============================================================================
pid = raw_dmm['pid']

all_corpus_list = []   

start_time = datetime.datetime.now()
for pid_num in list(pid):
    
    pid_corpus = []

    for code_idx in raw_dmm.columns[1:]:
        if code_idx=='P16':
            continue
        col_idx = raw_dmm.loc[raw_dmm.index[raw_dmm['pid']==pid_num], code_idx].values
        pid_corpus.extend(list(itertools.repeat(code_dic[code_idx], col_idx[0])))        

    all_corpus_list.append(pid_corpus)
    
    if (np.where(pid==pid_num)[0][0]+1)%10 == 0:
        print('pid', pid_num, round((np.where(pid==pid_num)[0][0]+1)/len(pid)*100, 2), '% done!' )
    
    
end_time = datetime.datetime.now()
print('Run time :', end_time-start_time)
# Run time : 0:06:13.176600


#%%
# pickle file save
with open('0711_2_AllCorpus.pkl', 'wb') as f:
    pickle.dump(all_corpus_list, f)




    
    
