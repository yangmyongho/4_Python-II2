# -*- coding: utf-8 -*-
"""
문2) 아래와 같은 단계로 kMeans 알고리즘을 적용하여 확인적 군집분석을 수행하시오.

 <조건> 변수 설명 : tot_price : 총구매액, buy_count : 구매횟수, 
                   visit_count : 매장방문횟수, avg_price : 평균구매액

  단계1 : 3개 군집으로 군집화
 
  단계2: 원형데이터에 군집 예측치 추가
  
  단계3 : tot_price 변수와 가장 상관계수가 높은 변수로 산점도(색상 : 클러스터 결과)
  
  단계4 : 산점도에 군집의 중심점 시각화
"""

import pandas as pd
from sklearn.cluster import KMeans # kMeans model
import matplotlib.pyplot as plt


sales = pd.read_csv("C:\\ITWILL\\Work\\4_Python-II\\data/product_sales.csv")
print(sales.info())
'''
RangeIndex: 150 entries, 0 to 149
Data columns (total 4 columns):
tot_price      150 non-null float64 -> 총구매금액 
visit_count    150 non-null float64 -> 매장방문수 
buy_count      150 non-null float64 -> 구매횟수 
avg_price      150 non-null float64 -> 평균구매금액 
'''



#  단계1 : 3개 군집으로 군집화
k = KMeans(n_clusters=3)
model = k.fit(sales)
model

pred = model.predict(sales)
pred  # 0 ~ 2




#  단계2: 원형데이터에 군집 예측치 추가
sales['cluster'] = pred
sales.info()
sales.head()




#  단계3 : tot_price 변수와 가장 상관계수가 높은 변수로 산점도(색상 : 클러스터 결과)
# 1) groupby로 상관관계 높은 변수 찾기
sales_grp = sales.groupby('cluster')
sales_grp.describe()
sales_grp.mean()  # avg_price, visit_count가 가장 상관계수 높은듯. 
'''
         tot_price  visit_count  buy_count  avg_price
cluster                                              
0         6.850000     2.071053   3.071053   5.742105
1         5.006000     0.244000   3.284000   1.464000
2         5.901613     1.433871   2.754839   4.393548
'''

# 2) 산점도
plt.scatter(x=sales['visit_count'], y=sales['avg_price'], c=sales['cluster'], marker='o')

# (+) 상관계수 correlation
sales.corr()  ## 이렇게 해도 결과 같음.
'''
             tot_price  visit_count  buy_count  avg_price   cluster
tot_price     1.000000     0.817954  -0.013051   0.871754 -0.349480
visit_count   0.817954     1.000000  -0.230612   0.962757 -0.203263
buy_count    -0.013051    -0.230612   1.000000  -0.278505 -0.333204
avg_price     0.871754     0.962757  -0.278505   1.000000 -0.170493
cluster      -0.349480    -0.203263  -0.333204  -0.170493  1.000000
'''


#  단계4 : 산점도에 군집의 중심점 시각화
centers = model.cluster_centers_
'''    tot_price    visit_count   buy_count   avg_price  
array([[6.85      , 2.07105263, 3.07105263, 5.74210526],
       [5.006     , 0.244     , 3.284     , 1.464     ],
       [5.9016129 , 1.43387097, 2.75483871, 4.39354839]])
'''
plt.scatter(centers[:,1], centers[:,3], c='red', marker='D', s=100)
























