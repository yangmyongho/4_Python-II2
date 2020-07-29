'''
 문) 중학교 1학년 신체검사(bodycheck.csv) 데이터 셋을 이용하여 다음과 같이 군집모델을 생성하시오.
  <조건1> 악력, 신장, 체중, 안경유무 칼럼 대상 [번호 칼럼 제외]
  <조건2> 계층적 군집분석의 완전연결방식 적용 
  <조건3> 덴드로그램 시각화 
  <조건4> 텐드로그램을 보고 3개 군집으로 서브셋 생성  
'''

import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram # 계층적 군집 model
import matplotlib.pyplot as plt

# data loading - 중학교 1학년 신체검사 결과 데이터 셋 
body = pd.read_csv("C:\\ITWILL\\Work\\4_Python-II\\data/bodycheck.csv", encoding='ms949')
print(body.info())
'''
 #   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   번호      15 non-null     int64
 1   악력      15 non-null     int64
 2   신장      15 non-null     int64
 3   체중      15 non-null     int64
 4   안경유무    15 non-null     int64'''
body.head()
'''
번호  악력   신장  체중  안경유무
0   1  28  146  34     1
1   2  46  169  57     2
2   3  39  160  48     2
3   4  25  156  38     1
4   5  34  161  47     1
'''





# <조건1> subset 생성 - 악력, 신장, 체중, 안경유무 칼럼  이용 
bodysub = body.iloc[:, 1:]
bodysub
type(bodysub)




# <조건2> 계층적 군집 분석  완전연결방식  
clusters = linkage(bodysub, method='complete', metric='euclidean')
clusters





# <조건3> 덴드로그램 시각화 : 군집수는 사용 결정 
plt.figure(figsize=(10,5))
dendrogram(clusters, leaf_font_size=20,leaf_rotation=90,)                
plt.show()                     
                     

                     

# <조건4> 텐드로그램을 보고 3개 군집으로 서브셋 생성
'''
cluster1 - 9 3 7 0 14
cluster2 - 10 2 4 5 13
cluster3 - 1 8 12 6 11
'''
from scipy.cluster.hierarchy import fcluster

cluster = fcluster(clusters, t=3, criterion='maxclust')
cluster  #  array([1, 3, 2, 1, 2, 2, 3, 1, 3, 1, 2, 3, 3, 2, 1], dtype=int32)
help(fcluster)


bodysub['cluster'] = cluster
cluster_grp = bodysub.groupby('cluster')
cluster_grp
cluster_grp.size()
'''
1    5
2    5
3    5
'''


# (+)군집별 특징

# 1) 산점도 시각화
plt.scatter(bodysub['신장'], bodysub['체중'], c=cluster, marker='o')

# 2) 군집별 통계
cluster_grp.mean()
'''
악력     신장    체중  안경유무
cluster                         
1        25.6  149.8  36.6   1.0
2        33.8  161.2  48.8   1.4
3        40.6  158.8  56.8   2.0
'''
cluster_grp.describe()
'''
           악력                              ... 안경유무                    
        count  mean       std   min   25%  ...  min  25%  50%  75%  max
cluster                                    ...                         
1         5.0  25.6  1.949359  23.0  25.0  ...  1.0  1.0  1.0  1.0  1.0
2         5.0  33.8  3.701351  29.0  32.0  ...  1.0  1.0  1.0  2.0  2.0
3         5.0  40.6  3.435113  38.0  38.0  ...  2.0  2.0  2.0  2.0  2.0
'''

















