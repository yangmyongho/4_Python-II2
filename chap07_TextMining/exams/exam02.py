# -*- coding: utf-8 -*-
'''
# 문) 2019년11월 ~ 2020년2월 까지(4개월) daum 뉴스기사를 다음과 같이 크롤링하고, 
     단어구름으로 시각화 하시오.
# <조건1> 날짜별 5개 페이지 크롤링
# <조건2> 불용어 전처리 
# <조건3> 단어빈도수 분석 후 top 20 단어 단어구름 시각화 
'''
import urllib.request as req # url 요청
from bs4 import BeautifulSoup # html 파싱
import pickle # -> list -> file save -> load(list)
import pandas as pd # 시계열 date
import re
from konlpy.tag import Kkma # class 
from re import match # 
from collections import Counter
from wordcloud import WordCloud # class
import matplotlib.pyplot as plt
import pickle



# 1. 수집년도 생성 : 시계열 date 이용
date = pd.date_range("2019-11-01", "2020-02-29")
len(date) # 121 (30+31+31+29)
sdate = [re.sub('-', '', str(d))[:8] for d in date]
sdate
len(sdate) # 121



# 2. Crawler 함수
def newsCrawler(date, pages=5) : # 1day news
    
    one_day_data =[]
    for page in range(1, pages+1): # 1~5page
        url = f"https://news.daum.net/newsbox?regDate={date}&page={page}"
        
        try:
            # url 요청
            res = req.urlopen(url)
            src = res.read() # source
            
            # html 파싱
            de_src = src.decode('utf-8') # 디코딩 <한글깨짐현상을 막아줌>
            html = BeautifulSoup(de_src, 'html.parser') # < html문서로 변경>
            
            # tag[속성='값'] -> "a[class='link_txt']"
            links = html.select("a[class='link_txt']")
        
            # 기사 내용만 추출
            one_page_data = [] # 빈 list
            
            print("date :", date)

            for link in links:
                link_str = str(link.string) # 내용만 추출후 문자타입으로 변경
                one_page_data.append(link_str.strip()) # 문장끝 불용어 처리(\n, 공백) , 빈 list에 삽입
            
            # 1day news
            one_day_data.extend(one_page_data[:40])
            #print(one_day_data)
        
        except Exception as e:
            print('오류 발생 :', e)
            
    return one_day_data



# 3. Crawler 함수 호출 
news_date_4month = [newsCrawler(date) for date in sdate]
news_date_4month
len(news_date_4month) # 121


# 3-1)중첩list -> 단일list
bendata = []
for sent in news_date_4month:
    bendata.extend(sent)
bendata
len(bendata) # 24200 (121*40*5) 



# 4. 명사 추출 : Kkma 사용
obj = Kkma() # object 생성

# 4-1) docs -> sentence
ex_sent = [obj.sentences(sent)[0] for sent in bendata] # 중첩list -> 단일list로 변경
ex_sent
len(ex_sent) # 24200


# 4-2) 명사 추출 (word count 가능하게)
nouns_word = []
for sent in ex_sent : 
    for noun in obj.nouns(sent): 
        nouns_word.append(noun)
nouns_word
len(nouns_word) # 236337



# 5. 전처리 + 단어카운트 : 단어 길이(1음절)제한, 숫자 제외
nouns_count = {} # 단어 카운트
for noun in nouns_word:
    if len(noun) > 1 and not(match('^[0-9]', noun)) :
        # key[noun] = value[출현빈도수]
        nouns_count[noun] = nouns_count.get(noun, 0) + 1
nouns_count
len(nouns_count) # 19143



# 6. WordCloud

# 6-1) top50 word
word_count = Counter(nouns_count) # dict
top20_word = word_count.most_common(20)
top20_word 


# 6-2) wordcloud
wc = WordCloud(font_path='C:/Windows/Fonts/malgun.ttf', width=800, height=600,
               max_words=100, max_font_size=200, background_color='white')
wc_result = wc.generate_from_frequencies(dict(top20_word))
#wc_result # <wordcloud.wordcloud.WordCloud at 0x1e5dc3f5a48>
plt.figure(figsize=(12,8))
plt.imshow(wc_result)
plt.axis('off') # x축,y축 테두리 제거
plt.show()


























































































