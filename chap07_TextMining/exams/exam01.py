'''
 문) 아래 url을 이용하여 어린이날(20200505)에 제공된 뉴스 기사를 
   1~5페이지 크롤링하는 크롤러 함수를 정의하고 크롤링 결과를 확인하시오.
   
   base_url = "https://news.daum.net/newsbox?regDate="   
   
   <조건1> 크롤러 함수의 파라미터(page번호, 날짜)
   <조건2> 크롤링 대상  : <a> 태그의 'class=link_txt' 속성을 갖는 내용 
   <조건3> 크롤링 결과 확인  : news 갯수와  문장 출력(한글 깨짐 확인)  
   http://news.daum.net/newsbox?regDate=20200505&page=1
'''

import urllib.request as req  # url 가져오기 
from bs4 import BeautifulSoup

base_url = "https://news.daum.net/newsbox?regDate="
'''
date1 = 20200505
page1 = range(1, 6)
len(page1) # 5
page = 5

day_news = base_url + str(date1)
day_news

pages = list(['&page=' + str(i) for i in range(1,6)])
pages

day_url = base_url + str(date1)
pages = list(['&page=' + str(i) for i in range(1, page+1)])
url = []
for page in pages:
    url.append(day_url+page)
url
'''

# 클로러 함수(페이지, 검색날자) 
def crawler_func(page, date):
    crawling_news = []
    url = base_url + str(date)
    for pg in range(1, page+1):
        final_url = url + '&page=' + str(pg)
        print(final_url)
    
        # url 요청
        res = req.urlopen(final_url)
        src = res.read() # source
        
        # html 파싱
        de_src = src.decode('utf-8') # 디코딩 <한글깨짐현상을 막아줌>
        html = BeautifulSoup(de_src, 'html.parser') # < html문서로 변경>
        
        # tag[속성='값'] -> "a[class='link_txt']"
        links = html.select("a[class='link_txt']")
    
        # 기사 내용만 추출
        one_page_data = [] # 빈 list
        
        for link in links:
            link_str = str(link.string) # 내용만 추출후 문자타입으로 변경
            one_page_data.append(link_str.strip()) # 문장끝 불용어 처리(\n, 공백) , 빈 list에 삽입
    
        crawling_news.extend(one_page_data[:40]) # 40개 선택 
            
    return crawling_news # 1~5 page news
    


# 클로러 함수 호출 
crawling_news = crawler_func(5, '20200505')

# 크롤링 결과 확인
print('크롤링 news 수 =', len(crawling_news)) 
# 크롤링 news 수 = 200
print(crawling_news)
    








































