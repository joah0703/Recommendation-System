# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 14:17:52 2022

@author: Yurim
"""

import numpy as np
import pandas as pd
from collections import Counter
from itertools import combinations
import math


# =============================================================================
# -- read SAMC output(z, theta)
# =============================================================================
def read_output(filename, data_type):
    with open(filename) as f:
        output = []
        out = f.readlines()
        for line in out:
            output.append(line.split())
            
    output = [list(map(data_type,i)) for i in output]
          
    return output


# =======================================
# -- for each topic, find ratio that each word carries that topic    
# =======================================
def word_dist_per_topic(word_topic_corpus, topic_in_corpus, topic_index):
    # for each topic, find ratio that each word carries that topic
    list_of_words = list(word_topic_corpus.keys())

    word_dist = [float((list(word_topic_corpus[word])[topic_index])/list(topic_in_corpus)[topic_index]) if list(topic_in_corpus)[topic_index] != 0 else 0
                 for word in list_of_words]
    # get top 10
    word_dist_rank_index = np.argsort(word_dist).tolist()[-10:] # since they sort by increasing order
    word = [list(word_topic_corpus.keys())[i] for i in word_dist_rank_index]
    percent = [word_dist[j] for j in word_dist_rank_index]
    
    dictionary = dict(zip(word, percent))
    return dictionary


# =======================================
# -- perpelxity
# =======================================      
def perplexity_r(theta_matrix, beta_matrix, corpus):
    n = 0; ll = 0.0
    
    for i1, text1 in enumerate(corpus):
        for word1 in text1:
            ll = ll + np.log((theta_matrix[i1]*np.array(beta_matrix[word1])).sum()) #text1.count(word1)*
            n = n + 1
    fy = np.exp(-ll/n)
    
    return fy


# =======================================
# -- coherence
# =======================================
def coherence_r(word_topic_corpus_r, topic_in_corpus_r, no_topic, corpus):
    result_topic = [word_dist_per_topic(word_topic_corpus_r, topic_in_corpus_r, topic_index) for topic_index in range(no_topic)]
                
    cx = Counter(); cxy = Counter()
    for text2 in corpus:
        for x in text2:
            cx[x] += 1
        for x, y in map(sorted, combinations(text2, 2)): #한 문장 내에서 단어들의 조합
            cxy[(x, y)] += 1
        
    topic_word=[]        
    for c, topic_w in enumerate(result_topic):
        topic_word.append(list(topic_w.keys())) #(result_topic)주제 내 단어만 추출(상위 10개)
             
    word_cb=[]; topic_pmi=[]
    for tt in range(no_topic):
        #tt=0
        word_cb_temp=[]
        for xx, yy in map(sorted, combinations(topic_word[tt], 2)):
            word_cb_temp.append((xx,yy,cxy[(xx,yy)])) #단어x, y, xy 조합 개수
        word_cb.append(word_cb_temp)

        w_pmi=[]
        pmi_avg = 0
        for c in range(len(word_cb[0])): #c=1. 단어xy의 조합 개수만큼 반복
            w1 = word_cb[tt][c][0] #첫번째 단어
            w2 = word_cb[tt][c][1] #두번째 단어
            if (cxy[(w1,w2)]>0)&(cx[w1]>0)&(cx[w2]>0):
                pmi = np.log(cxy[(w1,w2)]/sum(cx.values())**2)-np.log(cx[w1]/sum(cx.values()))-np.log(cx[w2]/sum(cx.values()))#np.log(cxy[(w1,w2)]/sum(cxy.values()))-np.log(cx[w1]/sum(cx.values()))-np.log(cx[w2]/sum(cx.values())) #-np.log(cxy[(w1,w2)]/(cx[w1]*cx[w2]))
                if pmi == math.inf:
                    print('pmi is inf')
                    pmi = 10.0**(8)
                if pmi == -math.inf: #추가
                    print('pmi is -inf')
                    pmi = -10.0**(8) #추가
                w_pmi.append(pmi)
                pmi_avg=np.mean(w_pmi)
            
        if pmi_avg!=0:
            topic_pmi.append(pmi_avg) #np.mean(topic_pmi)
            fx = np.mean(topic_pmi)      
    
    return fx
        
        
# =============================================================================
# -- topic_in_document_r
# =============================================================================
def topic_in_document_r_f(mycorpus, no_topic, samc_z):
    start_idx = 0
    topic_in_document_r = []
    
    for corpus1 in mycorpus:
        doc_per_topic = [0 for i in range(no_topic)]
        
        cor_topic_cnt = np.unique(samc_z[start_idx:start_idx+len(corpus1)], return_counts=True)
        start_idx += len(corpus1)
        
        for idx,loca in enumerate(cor_topic_cnt[0].tolist()):
            doc_per_topic[loca-1] += cor_topic_cnt[1].tolist()[idx]
        topic_in_document_r.append(doc_per_topic)

    return topic_in_document_r


# =============================================================================
# -- topic_in_corpus_r
# =============================================================================
def topic_in_corpus_r_f(topic_in_document_r):
    out = pd.DataFrame(topic_in_document_r).sum(axis=0).tolist()
    
    return out


# =============================================================================
# -- word_topic_corpus_r
# =============================================================================
def word_topic_corpus_r_f(z_w_df, V_len, no_topic):
    word_topic_corpus_r = {}
    sum_check = 0
    
    for i in range(V_len):
        word_per_topic = [0 for i in range(no_topic)]
    
        z_w_cnt = z_w_df.loc[z_w_df['w_idx_input']==i, 'samc_z'].value_counts()
        for idx, val in zip(z_w_cnt.index,z_w_cnt.values):
            word_per_topic[idx-1] += val
            
        word_topic_corpus_r[i] = word_per_topic
        sum_check += sum(word_per_topic)
        
    print(sum_check)
    return word_topic_corpus_r


# =============================================================================
# -- generating beta matrix by z
# =============================================================================
def samc_beta_m(result_samc, no_topic, V_len):
    samc_beta = pd.DataFrame(0, index=range(1,no_topic+1), columns=range(0,V_len))
    
    for topic in range(1,no_topic+1):
        for word in range(0,V_len):
            samc_beta.loc[topic,word] += len(result_samc[(result_samc['samc_z']==topic)&(result_samc['w_idx_input']==word)])
    samc_beta = samc_beta.div(samc_beta.sum(axis=1), axis=0) #행의 합으로 모든 값 나누기
    samc_beta = samc_beta.fillna(0.0) #모든 행이 0인 경우 nan
    samc_beta.sum(axis=1)
    # samc_beta.columns = gensim_beta.columns
    
    return samc_beta


#%%
# =============================================================================
# -- main
# =============================================================================
import os
os.chdir(r'E:\2022\0_study\2_추천시스템\0_유림_석사논문 작성\BLDA 유도\미디어패널')

##### corpus_idx, samc_z, samc_theta data should be seperated by '\n' for each document
corpus_idx = read_output('doc_word_0928.txt', int)
samc_z = read_output(r'C:\Users\Yurim\Downloads\z (1).txt', int)
samc_theta = read_output(r'C:\Users\Yurim\Downloads\theta (1).txt', float)

no_topic = 5
V_len = 113

w_flatten = sum(corpus_idx, [])
z_flatten = sum(samc_z, [])
z_word_idx_df = pd.DataFrame({'samc_z':z_flatten, 'w_idx_input':w_flatten})

samc_beta = samc_beta_m(z_word_idx_df, no_topic, V_len)
word_topic_corpus_r = word_topic_corpus_r_f(z_word_idx_df, V_len, no_topic)
topic_in_document_r = topic_in_document_r_f(corpus_idx, no_topic, z_flatten)
topic_in_corpus_r = topic_in_corpus_r_f(topic_in_document_r)



## perplexity score
print(perplexity_r(samc_theta, samc_beta, corpus_idx))
## coherence score
print(coherence_r(word_topic_corpus_r, topic_in_corpus_r, no_topic, corpus_idx))





#%%
## corpus 당 단거 개수 count
# N_list = []
# for cor in corpus_idx:
#     N_list.append(len(cor))

# with open('N_list.txt', 'w') as f:
#     f.writelines('%s ' %str(i) for i in N_list)
# =============================================================================
# gensim lda
# =============================================================================
from gensim.models.ldamodel import LdaModel
from gensim import corpora
import pandas as pd
import numpy as np
import os
import pickle
import json
# import warnings
# import time
import scipy as sp
from scipy.sparse import csr_matrix
# import pyLDAvis.gensim_models as gensimvis
# import pyLDAvis
import logging
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter

os.chdir(r'E:\2022\0_study\2_추천시스템\0_유림_석사논문 작성\BLDA 유도\미디어패널')

# =============================================================================
# read_data
# =============================================================================
with open('0711_2_AllCorpus.pkl', 'rb') as f:
    mycorpus = pickle.load(f)

with open('LDA_corpus.pkl', 'rb') as f:
    widx_num = pickle.load(f)

with open('LDA_dictionary.pkl', 'rb') as f:
    w_dic = pickle.load(f)


w_idx_input = []
for doc in widx_num:
    for word in doc:
        w_idx_input.extend([word[0] for i in range(word[1])])


#### lda input data 생성 (문서 별)
doc_word_idx = []
st_idx = 0
for i in mycorpus:
    doc_word_idx.append(w_idx_input[st_idx:st_idx+len(i)])
    st_idx += len(i)
    
    
    
# =============================================================================
# gensim LDA
# =============================================================================
no_topic = 5

dictionary = corpora.Dictionary(mycorpus)
corpus = [dictionary.doc2bow(text) for text in mycorpus]
# dictionary.token2id #w_dic과 동일한 순서

for iteration in range(10):
    ### LDA
    ldamodel = LdaModel(corpus, num_topics=no_topic, id2word=dictionary) #, passes=15, iterations=300
    gensim_beta = pd.DataFrame(ldamodel.get_topics(), columns=range(len(dictionary.token2id.keys())))
    
    
    word_topic = dict(zip(range(len(gensim_beta.columns)),gensim_beta.idxmax().tolist()))
    gensim_z = []
    for c in range(len(corpus)):
        z_list = []
        for i in range(len(corpus_idx[c])):
            z_list.append(word_topic[corpus_idx[c][i]])
        gensim_z.append(z_list)
    
    gensim_theta = []
    for c in range(len(corpus)):
        theta_m = [0 for i in range(no_topic)]
        for i,prob in dict(ldamodel[corpus[c]]).items():
            theta_m[i] = prob
        gensim_theta.append(theta_m)
        
    
    w_flatten = sum(corpus_idx, [])
    z_flatten = sum(gensim_z, [])
    
    z_word_idx_df = pd.DataFrame({'samc_z':z_flatten, 'w_idx_input':w_flatten})
    
    
    word_topic_corpus_r = word_topic_corpus_r_f(z_word_idx_df, V_len, no_topic)
    topic_in_document_r = topic_in_document_r_f(corpus_idx, no_topic, z_flatten)
    topic_in_corpus_r = topic_in_corpus_r_f(topic_in_document_r)
    
    
    
    ## perplexity score
    print(perplexity_r(gensim_theta, gensim_beta, corpus_idx))
    ## coherence score
    print(coherence_r(word_topic_corpus_r, topic_in_corpus_r, no_topic, corpus_idx))
    
    
    ## 결과 저장
    gensim_beta.to_csv('221001 media gensim result/gensim_beta'+str(iteration)+'.csv')
            
    with open('221001 media gensim result/gensim_theta'+str(iteration)+'.txt', 'w') as f:
        for doc in gensim_theta:
            f.writelines('%s ' %str(i) for i in doc)
            f.write('\n')
