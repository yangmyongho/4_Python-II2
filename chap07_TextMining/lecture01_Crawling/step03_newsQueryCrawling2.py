# -*- coding: utf-8 -*-
"""
step03_newsQueryCrawling2.py

방법 2) url query 이용 : 년도별 뉴스 자료 수집
        ex) date : 2015.01.01 ~ 2020.01.01 (60개월)
            page : 1 ~ 5
        ex2) date : 2015.01.01 ~ 2015.03.01 
            page : 1 ~ 4

"""
import urllib.request as req # url 요청
from bs4 import BeautifulSoup # html 파싱
import pickle # -> list -> file save -> load(list)
import pandas as pd # 시계열 date
import re # sub('pattern', '', string)



'''
# 1. 수집 년도 생성 : 시계열 date 이용
date = pd.date_range("2015-01-01", "2020-01-01")
len(date) # 1827 =<365*3 + 366*2> 366=윤달
date[0] # Timestamp('2015-01-01 00:00:00', freq='D')
date[-1] # Timestamp('2020-01-01 00:00:00', freq='D')

# '2020-01-01 00:00:00'  ->  '20200101'
sdate = [re.sub('-', '', str(d))[:8] for d in date]
sdate[0] # 20150101
len(sdate) # 1827



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
        
        except Exception as e:
            print('오류 발생 :', e)
            
    return one_day_data



# 3. Crawler 함수 호출 
year5_news_date = [newsCrawler(date)[0] for date in sdate] # [0]을 넣는이유: 단일list로하기위해
year5_news_date
'''


# 1. 수집 년도 생성 : 시계열 date 이용
date = pd.date_range("2015-01-01", "2015-03-01")
len(date) 
date[0] # Timestamp('2015-01-01 00:00:00', freq='D')
date[-1] # Timestamp('2020-01-01 00:00:00', freq='D')

# '2020-01-01 00:00:00'  ->  '20200101'
sdate = [re.sub('-', '', str(d))[:8] for d in date]
sdate[0] # 20150101
len(sdate) # 1827



# 2. Crawler 함수
def newsCrawler(date, pages=5) : # 1day news
    
    one_day_data =[]
    for page in range(1, pages): # 1~5page
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
year5_news_date = [newsCrawler(date) for date in sdate]
year5_news_date
len(year5_news_date) # 60










































































