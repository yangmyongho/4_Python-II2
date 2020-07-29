# -*- coding: utf-8 -*-
"""
step03_news_wordVector

news crawling data -> word vector
    문장 -> 단어 벡터 -> 희소행렬(sparse matrix)
    ex) '직업은 데이터 분석가 입니다.' -> '직업', '데이터', '분석가'
"""
from konlpy.tag import Kkma # class 
from re import match # 
from collections import Counter
from wordcloud import WordCloud # class
import matplotlib.pyplot as plt
import pickle



# 1. pickle file 읽기 : news_data.pck
file = open('../data/news_data.pck', mode='rb')
news_data = pickle.load(file)
file.close()
news_data
type(news_data) # list
len(news_data) # 11600



# 2. docs -> sentence
obj = Kkma() # object 생성
ex_sent = [obj.sentences(sent)[0] for sent in news_data] # 중첩list -> 단일list로 변경
ex_sent
len(ex_sent) # 11600
ex_sent[0] # '의협 " 감염병 위기 경보 상향 제안.. 환자 혐오 멈춰야"'
ex_sent[-1] # '미, 인건비 우선 협의 제안에 " 포괄적 SMA 신속 타결 대단히 손상"'



# 3. sentence -> word vector
sentence_nouns = [] # 단어 벡터 저장
for sent in ex_sent : # 11600
    word_vec = ""
    for noun in obj.nouns(sent) : # 문장 -> 명사추출
        if len(noun) > 1 and not(match('^[0-9]', noun)) :
            word_vec += noun + " " # 공백을넣어서 누적하겠다.
    print(word_vec)
    sentence_nouns.append(word_vec)

len(sentence_nouns) # 11600
ex_sent[0] # '의협 " 감염병 위기 경보 상향 제안.. 환자 혐오 멈춰야"'
sentence_nouns[0] # '의협 감염병 위기 경보 상향 제안 환자 혐오 '
ex_sent[-1] # '미, 인건비 우선 협의 제안에 " 포괄적 SMA 신속 타결 대단히 손상"'
sentence_nouns[-1] # '인건비 우선 협의 제안 포괄적 신속 타결 손상 '



# 4. file save
file = open('../data/sentence_nouns.pickle', mode='wb')
pickle.dump(sentence_nouns, file)
print('file save')
file.close()


# 4-1) file load
file = open('../data/sentence_nouns.pickle', mode='rb')
word_vector = pickle.load(file)
word_vector
len(word_vector) # 11600






























