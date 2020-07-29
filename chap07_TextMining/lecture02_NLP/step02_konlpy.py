# -*- coding: utf-8 -*-
"""
step02_konlpy

Konlpy : 한글 형태소 분석을 제공하는 패키지
pip install konlpy

"""
import konlpy
from konlpy.tag import Kkma # class



# 1. 
obj = Kkma() # 생성자 -> object
para = "나는 홍길동 입니다. 나이는 23세 입니다. 대한민국 만세 입니다."

# 문단 -> 문장
ex_sent = obj.sentences(para)
ex_sent # ['나는 홍길동 입니다.', '나이는 23세 입니다.', '대한민국 만세 입니다.']
len(ex_sent) # 3

# 문단 -> 단어(명사)
ex_word = obj.nouns(para)
ex_word
len(ex_word) # 10

# 문단 -> 품사(형태소)
ex_pos = obj.pos(para)
ex_pos
type(ex_pos) # list [('word', '품사'), ('word', '품사'), ...]

# 지정한 품사만 추출
''' NNG 일반 명사 NNP 고유 명사 NNB 의존 명사 NR 수사 NP 대명사
   ㄴ> NNG 일반 명사 NNP 고유 명사 NP 대명사   '''
nouns = [] # 명사 저장
for word, wclass in ex_pos : # ('word', '품사')
    if wclass == 'NNG' or wclass == 'NNP' or wclass == 'NP' :
        nouns.append(word)
nouns # ['나', '홍길동', '나이', '대한민국', '만세']
















