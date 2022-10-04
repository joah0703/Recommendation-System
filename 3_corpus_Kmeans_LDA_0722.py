# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 11:25:23 2022

@author: Yurim

2021 미디어패널 LDA
https://wikidocs.net/30708
https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=pdc222&logNo=221360553916
"""

from gensim.models.ldamodel import LdaModel
from gensim import corpora
import pandas as pd
import numpy as np
import os
import pickle
# import warnings
# import time
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
from gensim.models.coherencemodel import CoherenceModel
import logging
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

os.chdir(r'E:\2022\0_study\2_추천시스템\220722')

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
SEED = 1

# read
with open('0711_2_AllCorpus.pkl', 'rb') as f:
    mycorpus = pickle.load(f)
    
#%%
# =============================================================================
# LDA 토픽 개수 정하기 - coherence 주제 일관성 지표
# =============================================================================
## ctrl+Enter로 실행하기!


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    perplexity_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        print('========== Training', num_topics, 'epochs ==========')
        
        model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=SEED)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        
        model_list.append(model)
        coherence_values.append(coherencemodel.get_coherence())
        perplexity_values.append(model.log_perplexity(corpus))
        

    return model_list, coherence_values, perplexity_values


def find_optimal_number_of_topics(dictionary, corpus, processed_data):
    limit = 11
    start = 2
    step = 1

    model_list, coherence_values, perplexity_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=processed_data, start=start, limit=limit, step=step)

    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()
    
    plt.plot(x, perplexity_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Perplexity score")
    plt.legend(("Perplexity_values"), loc='best')
    plt.show()
    

processed_data = mycorpus

# 정수 인코딩과 빈도수 생성
dictionary = corpora.Dictionary(mycorpus)
corpus = [dictionary.doc2bow(text) for text in mycorpus]
print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))



# 최적의 토픽 수 찾기
find_optimal_number_of_topics(dictionary, corpus, processed_data)



#%%
## corpus를 txt 파일로 저장
new_data = []
for i in corpus:
    new_data.append(','.join(str(e) for e in i))

with open('LDA_corpus.pkl','wb') as f:
    pickle.dump(new_data, f)
# with open('LDA_corpus.txt','w',encoding='UTF-8') as f:
#     for name in new_data:
#         f.write(name+'\n')
       
        
## dictionary를 txt 파일로 저장
dic_dt = dictionary.token2id

with open('LDA_dictionary.pkl','wb') as f:
    pickle.dump(dic_dt, f)
    # for code,name in dic_dt.items():
    #     f.write(f'{code} : {name}\n')

#%%
# =============================================================================
# 최적의 훈련 반복 횟수 찾기?
# =============================================================================
# coherences=[]
# perplexities=[]
# passes=[]
# warnings.filterwarnings('ignore')

# for i in range(10):
    
#     ntopics = 3
#     if i==0:
#         p=1
#     else:
#         p=i*5
#     tic = time.time()
#     lda4 = LdaModel(corpus, id2word=dictionary, num_topics=ntopics, iterations=400, passes=p, random_state=1)
#     print('epoch',p,time.time() - tic)
#     # iteration: 문서당 반복 횟수, passes: 전체 코퍼스 훈련 횟수
    
#     cm = CoherenceModel(model=lda4, corpus=corpus, coherence='u_mass')
#     coherence = cm.get_coherence()
#     print("Cpherence",coherence)
#     coherences.append(coherence)
#     print('Perplexity: ', lda4.log_perplexity(corpus),'\n\n')
#     perplexities.append(lda4.log_perplexity(corpus))



#%%
# =============================================================================
# LDA
# =============================================================================

# dictionary = corpora.Dictionary(mycorpus)
# corpus = [dictionary.doc2bow(text) for text in mycorpus]
# print(corpus[1])


NUM_TOPICS = 10 # 토픽 개수

ldamodel = LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, random_state=SEED, passes=15, iterations=400) #
# passes : 알고리즘의 동작 횟수 (알고리즘이 결정하는 토픽의 값이 적절히 수렴할 수 있도록 충분히 적당한 횟수)
#topics = ldamodel.print_topics(num_words=10) #단어 개수

topics = ldamodel.show_topics()

topic_df = pd.DataFrame(topics, columns=['num', 'topic'])
#topic_df.to_csv('0711_3_topic'+str(NUM_TOPICS)+'.csv', encoding='utf-8-sig')


############## 0313_topic10.csv 파일 포멧 변경
#raw_topic10 = pd.read_csv('0313_topic10.csv')
raw_topic10 = topic_df
raw_topic10.columns

#del raw_topic10['Unnamed: 0']
after = raw_topic10['topic'].apply(lambda x: x.split('"')[1::2])


new = pd.DataFrame(after[0])
for i in range(1,NUM_TOPICS):
    new = pd.concat([new, pd.DataFrame(after[i])], axis=1)

new2 = new.transpose().reset_index()
del new2['index']
topic10_text = pd.concat([raw_topic10, new2], axis=1)


topic10_text.to_csv('0711_3_topic'+str(NUM_TOPICS)+'_text.csv', encoding='utf-8-sig', index=False)



# =============================================================================
# 문서별 토픽 확인
# =============================================================================
def make_topictable_per_doc(ldamodel, corpus):
    topic_table = pd.DataFrame()

    # 몇 번째 문서인지를 의미하는 문서 번호와 해당 문서의 토픽 비중을 한 줄씩 꺼내온다.
    for i, topic_list in enumerate(ldamodel[corpus]):
        doc = topic_list[0] if ldamodel.per_word_topics else topic_list            
        doc = sorted(doc, key=lambda x: (x[1]), reverse=True)
        # 각 문서에 대해서 비중이 높은 토픽순으로 토픽을 정렬한다.

        # 모든 문서에 대해서 각각 아래를 수행
        for j, (topic_num, prop_topic) in enumerate(doc): #  몇 번 토픽인지와 비중을 나눠서 저장한다.
            if j == 0:  # 정렬을 한 상태이므로 가장 앞에 있는 것이 가장 비중이 높은 토픽
                topic_table = topic_table.append(pd.Series([int(topic_num), round(prop_topic,4), topic_list]), ignore_index=True)
                # 가장 비중이 높은 토픽과, 가장 비중이 높은 토픽의 비중과, 전체 토픽의 비중을 저장한다.
            else:
                break
    return(topic_table)


topictable = make_topictable_per_doc(ldamodel, corpus)
topictable = topictable.reset_index() # 문서 번호을 의미하는 열(column)로 사용하기 위해서 인덱스 열을 하나 더 만든다.
topictable.columns = ['문서 번호', '가장 비중이 높은 토픽', '가장 높은 토픽의 비중', '각 토픽의 비중']
#topictable[:10]

pid_num = pd.read_csv('0711_1_DMM.csv')['pid']
topictable.insert(1, 'pid', pid_num)

topictable.to_csv('0711_3_pid_topic'+str(NUM_TOPICS)+'.csv', encoding='utf-8-sig')




# LDA result visualization - HTML
lda_visualization = gensimvis.prepare(ldamodel, corpus, dictionary, sort_topics=False)
pyLDAvis.save_html(lda_visualization, 'LDA result'+str(NUM_TOPICS)+'.html')


#%%
# =============================================================================
# Kmeans
# =============================================================================
dictionary = corpora.Dictionary(mycorpus)
corpus = [dictionary.doc2bow(text) for text in mycorpus]

NUM_TOPICS = 5 # 토픽 개수


total_list = []
for i in range(len(corpus)):
    maxlist = [(x,0) for x in range(0,len(dictionary.token2id))]
    for j in corpus[i]:
        if j[0] in [x[0] for x in maxlist]:
            maxlist[j[0]] = j
    total_list.append(maxlist)

kdata = [pd.DataFrame(total_list[i])[1].tolist() for i in range(len(total_list))]
# pickle file save
with open('0722_kmeans_input.pkl', 'wb') as f:
    pickle.dump(kdata, f)

Kmodel = KMeans(n_clusters = NUM_TOPICS, random_state = SEED)
Kmodel.fit(kdata)
pred_group = Kmodel.predict(kdata)



# Save KMeans result
topictable  = pd.DataFrame(pred_group, columns=['k_group'])

pid_num = pd.read_csv('0711_1_DMM.csv')['pid']
topictable.insert(0, 'pid', pid_num)

topictable.to_csv('0722_kmeans'+str(NUM_TOPICS)+'.csv')




# visualization
# https://lovit.github.io/nlp/2018/09/27/pyldavis_kmeans/
import pyLDAvis
from kmeans_visualizer import kmeans_to_prepared_data

prepared_data = kmeans_to_prepared_data(x, index2word, centers, labels)
pyLDAvis.display(prepared_data)

# visualization KMeans result
# import matplotlib.pyplot as plt

# plt.figure(figsize = (8, 8))

# for i in range(NUM_TOPICS):
#     plt.scatter(topictable.loc[topictable['k_group'] == i, 'Annual Income (k$)'], df.loc[df['cluster'] == i, 'Spending Score (1-100)'], 
#                 label = 'cluster ' + str(i))

# plt.legend()
# plt.title('K = %d results'%k , size = 15)
# plt.xlabel('Annual Income', size = 12)
# plt.ylabel('Spending Score', size = 12)
# plt.show()


#%%
# =============================================================================
# 아래의 연관분석 코드는 필요하면 수정 필요
# =============================================================================








#%%
# =============================================================================
# apriori 연관분석
# https://m.blog.naver.com/eqfq1/221444712369
# =============================================================================
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori


te = TransactionEncoder()
te_ary = te.fit(mycorpus).transform(mycorpus)
# mycorpus : 맨 위에서 불러온 pickle 파일
df = pd.DataFrame(te_ary, columns=te.columns_) #위에서 나온걸 보기 좋게 데이터프레임으로 변경

frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)
frequent_itemsets 

from mlxtend.frequent_patterns import association_rules
associ_rules = pd.DataFrame(association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)) 

associ_rules.to_csv('0313_assiciation all corpus.csv', encoding='utf-8-sig')


#%%


#%%
# =============================================================================
# 토픽별 문서 인덱스
# =============================================================================
topictable = pd.read_csv('0313_pid_topic.csv')
topictable.columns

topic_dic = {}

for topic_num in range(NUM_TOPICS):
    topic_dic[topic_num] = topictable.loc[topictable['가장 비중이 높은 토픽']==topic_num, '문서 번호'].tolist()


# =============================================================================
# 토픽 별 apriori 연관분석
# https://m.blog.naver.com/eqfq1/221444712369
# =============================================================================
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori


for j in range(NUM_TOPICS):
    globals()['topic_corpus'+str(j)] = []
    for i in topic_dic[j]:
        globals()['topic_corpus'+str(j)].append(mycorpus[i])
        
       
for i in range(NUM_TOPICS):
    te = TransactionEncoder()
    te_ary = te.fit(globals()['topic_corpus'+str(i)]).transform(globals()['topic_corpus'+str(i)])
    # mycorpus : 맨 위에서 불러온 pickle 파일
    df = pd.DataFrame(te_ary, columns=te.columns_) #위에서 나온걸 보기 좋게 데이터프레임으로 변경
    
    frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)
    frequent_itemsets 
    
    from mlxtend.frequent_patterns import association_rules
    associ_rules = pd.DataFrame(association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)) 
    
    associ_rules.to_csv('0313_assiciation per topic'+str(i)+'.csv', encoding='utf-8-sig')


