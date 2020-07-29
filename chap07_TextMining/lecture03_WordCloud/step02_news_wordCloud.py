# -*- coding: utf-8 -*-
"""
step02_news_wordCloud


1. text file 읽기 : news crawling data
2. 명사 추출 : Kkma 사용
3. 전처리 + 단어카운트 : 단어 길이(1음절)제한, 숫자 제외
4. WordCloud
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



# 2. 명사 추출 : Kkma 사용
obj = Kkma() # object 생성

# 2-1) docs -> sentence
ex_sent = [obj.sentences(sent)[0] for sent in news_data] # 중첩list -> 단일list로 변경
ex_sent
len(ex_sent) # 11600


# 2-2) 명사 추출 (word count 가능하게)
nouns_word = []
for sent in ex_sent : 
    for noun in obj.nouns(sent): 
        nouns_word.append(noun)
nouns_word
len(nouns_word) # 120939



# 3. 전처리 + 단어카운트 : 단어 길이(1음절)제한, 숫자 제외
nouns_count = {} # 단어 카운트
for noun in nouns_word:
    if len(noun) > 1 and not(match('^[0-9]', noun)) :
        # key[noun] = value[출현빈도수]
        nouns_count[noun] = nouns_count.get(noun, 0) + 1
nouns_count
len(nouns_count) # 12194

# word cloud -> 수작업
nouns_count['확진자'] = nouns_count['진자']
del nouns_count['진자']

# 4. WordCloud

# 4-1) top50 word
word_count = Counter(nouns_count) # dict
top50_word = word_count.most_common(50)
top50_word 
'''
[('코로나', 2539), ('종합', 2008), ('신종', 659), ('확진자', 626), ('중국', 554),
 ('환자', 536), ('정부', 402), ('한국', 383), ('대구', 365), ('감염', 362),
 ('격리', 360), ('신천지', 322), ('마스크', 312), ('추가', 311), ('확산', 305),
 ('병원', 302), ('단독', 278), ('입국', 244), ('검사', 234), ('통합', 226),
 ('번째', 219), ('대통령', 213), ('사망', 212), ('발생', 203), ('지역', 198),
 ('조사', 196), ('일본', 194), ('우려', 190), ('서울', 187), ('국내', 184),
 ('크루즈', 180), ('트럼프', 167), ('방문', 164), ('교민', 161), ('금지', 159),
 ('공천', 158), ('대응', 158), ('미래', 156), ('검찰', 153), ('검토', 147),
 ('경찰', 145), ('제주', 143), ('비상', 143), ('총선', 143), ('중단', 142),
 ('수사', 141), ('전국', 139), ('날씨', 138), ('논란', 137), ('사망자', 137)] '''

# 4-2) wordcloud
wc = WordCloud(font_path='C:/Windows/Fonts/malgun.ttf', width=800, height=600,
               max_words=100, max_font_size=200, background_color='white')
wc_result = wc.generate_from_frequencies(dict(top50_word))
#wc_result # <wordcloud.wordcloud.WordCloud at 0x1e5dc3f5a48>
plt.figure(figsize=(12,8))
plt.imshow(wc_result)
plt.axis('off') # x축,y축 테두리 제거
plt.show()

