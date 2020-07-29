# -*- coding: utf-8 -*-
"""
step02_newsQueryCrawling

news Crawling : url query 이용
    url : http://media.daum.net -> [배열 이력]
    url : http://news.daum.net/newsbox -> base url -> [특정 날짜]
    url : https://news.daum.net/newsbox?regDate=20200505 -> [특정 페이지] 
    url : https://news.daum.net/newsbox?regDate=20200505&page=2
    
사용자 url : http://news.daum.net/newsbox?regDate=20200201&page=1
"""
import urllib.request as req # url 요청
from bs4 import BeautifulSoup # html 파싱
import pickle # -> list -> file save -> load(list)



# 1. url query 만들기
'''
date : 2020.02.01 ~ 2020.02.29
page : 1 ~ 10 (page)
'''

# 1-1) base url
url = "http://news.daum.net/newsbox" 
base_url = "http://news.daum.net/newsbox?regDate=" # '?regDate=' 추가


# 1-2) date 결합
date = list(range(20200201, 20200230))
len(date) # 29

url_list = [base_url + str(d) for d in date]
url_list
# 'http://news.daum.net/newsbox?regDate=20200201'
# 'http://news.daum.net/newsbox?regDate=20200202'


# 1-3) page 결합
page1 = list(range(1, 11)) # 1 ~ 10
len(page1) # 10
pages = ['&page='+str(p) for p in page1] # &page=1 ~ &page=10 

final_url = []
for url in url_list : # base url + date
    for page in pages : # page1 ~ page10
        final_url.append(url + page)
        # https://news.daum.net/newsbox?regDate=20200505&page=1
       
        
# 1-4) 결과 확인
len(final_url) # 290
final_url[0] # 'http://news.daum.net/newsbox?regDate=20200201&page=1'
final_url[-1] # 'http://news.daum.net/newsbox?regDate=20200229&page=10'



# 2. Crawler 함수 정의 

# 2-1) 1page추출 함수 생성 
def Crawler(url):
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
    cnt = 0
    
    for link in links:
        link_str = str(link.string) # 내용만 추출후 문자타입으로 변경
        cnt += 1
        #print('cnt :', cnt)  <확인했으니까 생략함>
        #print('news :', link_str)  <확인했으니까 생략함>
        one_page_data.append(link_str.strip()) # 문장끝 불용어 처리(\n, 공백) , 빈 list에 삽입

    return one_page_data[:40] # 이렇게하면 필요한 기사만 추출


# 2-2) 1page추출 함수 확인 
first_data = Crawler(final_url[0]) # 2020.02.01 page1 기사 크롤링
len(first_data) # 134
type(first_data) # list
first_data[0] # '의협 "감염병 위기경보 상향 제안..환자 혐오 멈춰야"'
first_data[-1] # "'공인구 합격이라는데' 왜 홈런은 펑펑 터질까"
# 문제점 : 오른쪽에 링크된 실시간 뉴스 까지 같이 나옴 
# 문제점 해결 : 해당 기사만 추출
#first_data[:40] # 특정날짜 특정페이지 기사만 추출



# 3. Crawler 함수 호출

# 3-1) 2월 (1개월) 전체 news 수집
month_news = []
page_cnt = 0

for url in final_url:
    page_cnt += 1
    print(page_cnt,'페이지')
    one_page_news = Crawler(url) # 1page news
    try: # 예외처리 : url 없는경우
        print('one page news')
        print(one_page_news)
        # 중첩 vs 단일 선택 (단일)
        #month_news.append(one_page_news) # [[1page], [2page], ...] <중첩list>
        month_news.extend(one_page_news) # [1page, 2page, ...] <단일list>
    except :
        print('해당 url 없음 :', url)


# 3-2) 결과 확인
month_news
len(date) # 29 <날짜 총수>
len(page1) # 10 <페이지 총수>
len(first_data) # 40 <1page에 있는 기사 수>
len(final_url) # 290 =<29 * 10> 
len(month_news) # 11600 =<290 * 40>



# 4. binary file save
# 상대경로 : C:\ITWILL\4_Python-II\workspace\chap07_TextMining\lecture01_Crawling

# 4-1) file save
file = open('../data/news_data.pck', mode='wb')
pickle.dump(month_news, file)
file.close()


# 4-2) file load
file = open('../data/news_data.pck', mode='rb')
month_news2 = pickle.load(file)
month_news2 # 
file.close()
month_news[0] # '의협 "감염병 위기경보 상향 제안..환자 혐오 멈춰야"'
month_news2[0] # '의협 "감염병 위기경보 상향 제안..환자 혐오 멈춰야"'
 # 위아래 동일
































