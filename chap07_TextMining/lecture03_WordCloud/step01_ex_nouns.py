# -*- coding: utf-8 -*-
"""
step01_ex_nouns

1. text file 읽기
2. 명사 추출 : Kkma 사용
3. 전처리 + 단어카운트 : 단어 길이(1음절)제한, 숫자 제외
4. WordCloud

"""
from konlpy.tag import Kkma # class 
from re import match # 
from collections import Counter
from wordcloud import WordCloud # class
import matplotlib.pyplot as plt



# 1. text file 읽기 : 
file = open('../data/text_data.txt', mode='r', encoding='utf-8')
docs = file.read()
docs
''' '형태소 분석을 시작합니다. 나는 데이터 분석을 좋아합니다. 
     \n직업은 데이터 분석 전문가 입니다. Text mining 기법은 2000대 초반에 개발된 기술이다.'   '''



# 2. 명사 추출 : Kkma 사용
obj = Kkma() # object 생성

# 2-1) docs -> sentence
ex_sent = obj.sentences(docs)
ex_sent
'''
['형태소 분석을 시작합니다.',
 '나는 데이터 분석을 좋아합니다.',
 '직업은 데이터 분석 전문가 입니다.',
 'Text mining 기법은 2000대 초반에 개발된 기술이다.'] '''
len(ex_sent) # 4


# 2-2) docs -> nouns <여기서 추출된 명사는 중복하지않으므로 카운트 할수없다.>
ex_nouns = obj.nouns(docs) # 유일한 명사 추출
ex_nouns
'''['형태소', '분석', '나', '데이터', '직업', '전문가', '기법', '2000', 
    '2000대', '대', '초반', '개발', '기술']   '''
len(ex_nouns) # 13


# 2-3) 명사 추출 (word count 가능하게)
nouns_word = []
for sent in ex_sent : # '형태소 분석을 시작합니다.'
    for noun in obj.nouns(sent): # 문장 -> 명사
        nouns_word.append(noun)
nouns_word
''' ['형태소', '분석', '데이터', '분석', '직업', '데이터', '분석', '전문가',
     '기법', '2000', '2000대', '대', '초반', '개발', '기술']   '''
len(nouns_word) # 15



# 3. 전처리 + 단어카운트 : 단어 길이(1음절)제한, 숫자 제외
nouns_count = {} # 단어 카운트
for noun in nouns_word:
    if len(noun) > 1 and not(match('^[0-9]', noun)) :
        # key[noun] = value[출현빈도수]
        nouns_count[noun] = nouns_count.get(noun, 0) + 1
nouns_count
''' {'형태소': 1, '분석': 3, '데이터': 2, '직업': 1, '전문가': 1,
     '기법': 1, '초반': 1, '개발': 1, '기술': 1}   '''
len(nouns_count) # 9

    

# 4. WordCloud

# 4-1) top5 word
word_count = Counter(nouns_count) # dict
top5_word = word_count.most_common(5)
top5_word # [('분석', 3), ('데이터', 2), ('형태소', 1), ('직업', 1), ('전문가', 1)]


# 4-2) wordcloud
wc = WordCloud(font_path='C:/Windows/Fonts/malgun.ttf', width=500, height=400,
               max_words=100, max_font_size=150, background_color='white')
wc_result = wc.generate_from_frequencies(dict(top5_word))
wc_result # <wordcloud.wordcloud.WordCloud at 0x1e5dc3f5a48>
plt.imshow(wc_result)
plt.axis('off') # x축,y축 테두리 제거
plt.show()

wc_result2 = wc.generate_from_frequencies(dict(nouns_count))
plt.imshow(wc_result2)
plt.axis('off') # x축,y축 테두리 제거
plt.show()




























