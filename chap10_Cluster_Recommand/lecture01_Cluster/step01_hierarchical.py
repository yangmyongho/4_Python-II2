# -*- coding: utf-8 -*-
"""
계층적 군집분석 
 - 상향식(Bottom=up)으로 계층적 군집을 형성해나감.
 - 군집 형성 기준 : 유클리디안 거리계산식 이용
 - 숫자형 변수만 사용할 수 있다.(유클리디안 거리계산식이용하기 위해)
"""

import pandas as pd  # dataframe 생성
from sklearn.datasets import load_iris  # 5째 컬럼이 숫자로 되어있는.
from scipy.cluster.hierarchy import linkage, dendrogram  # 군집분석 관련 tool
import matplotlib.pyplot as plt  # 군집분석 결과를 산점도로 시각화



# 1. dataset load
iris = load_iris()
iris.feature_names

X = iris.data
X.shape  # (150, 4)
y = iris.target
y  # 0 ~ 2

iris_df = pd.DataFrame(X, columns=iris.feature_names)
sp = pd.Series(y)
sp

# DataFrame에 y변수 추가
iris_df['species'] = sp

iris_df.info()
'''
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   sepal length (cm)  150 non-null    float64
 1   sepal width (cm)   150 non-null    float64
 2   petal length (cm)  150 non-null    float64
 3   petal width (cm)   150 non-null    float64
 4   species            150 non-null    int32  
'''



# 2. 계층적 군집분석
clusters = linkage(y = iris_df, method = 'complete', metric = 'euclidean')
'''
method = 'single'  :  단순연결
method = 'complete'  :  완전연결
method = 'average'  :  평균연결
'''
clusters.shape  # (149, 4)  >> 클러스터를 생성할 때마다 하나씩 수가 줄어드는것은 1번째 관측치를 기준으로 나머지 관측치를,
                #              두번째 관측치를 기준으로 나머지 관측치에 대한 거리를 계산하기 때문
                #              유클리디안 거리 계산 결과는 이등변삼각형 모양으로 나타남.

clusters
'''
array([[1.01000000e+02, 1.42000000e+02, 0.00000000e+00, 2.00000000e+00],
       [7.00000000e+00, 3.90000000e+01, 1.00000000e-01, 2.00000000e+00],
       .
       .
       .]
'''



# 3. 덴드로그램 시각화
plt.figure(figsize=(20,5))
dendrogram(Z=clusters, leaf_rotation=90, leaf_font_size=20,)  # 끝에 쉼표 찍어줄 것.
plt.show()




# 4. 클러스터 자르기/평가 : 3단계 덴드로그램 결과로 판단한다
from scipy.cluster.hierarchy import fcluster  # cluster 자르기

# 1) 클러스터 자르기
cluster = fcluster(clusters, t=3, criterion='distance')
# t는 자를 클러스터 수, criterion은 자르는 기준(distance는 유클리디안 거리)
cluster
'''
array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
       3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
       3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 2, 2, 2, 2, 3, 2, 2, 2,
       2, 3, 2, 3, 3, 2, 2, 2, 2, 3, 2, 3, 2, 3, 2, 2, 3, 3, 2, 2, 2, 2,
       2, 3, 3, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 3], dtype=int32)
'''

# 2) DF 칼럼 추가
iris_df['cluster'] = cluster
iris_df.info()
iris_df.head()
iris_df.tail()

# 3) 산점도 시각화 -> 1번째 칼럼을 x축, 3번째 컬럼을 y축으로, cluster별 색상지정하려고 함.
plt.scatter(x=iris_df['sepal length (cm)'], y=iris_df['petal length (cm)'],
            c=iris_df['cluster'], marker='o')

# 4) 관측치(species) vs 예측치(cluster라고 치자)
cross = pd.crosstab(iris_df['species'], iris_df['cluster'])
cross
'''
cluster   1   2   3
species            
0        50   0   0
1         0   0  50
2         0  34  16
'''

# 5) 군집별 특성분석
# DF.groupby('집단변수')

cluster_grp = iris_df.groupby('cluster')
cluster_grp.size()
'''
1    50
2    34
3    66
dtype: int64
'''
cluster_grp.mean()
'''     <-------------------- 집단들 ----------------------------->    집단
         sepal length (cm)  sepal width (cm)  ...  petal width (cm)   species
cluster                                       ...                            
1                 5.006000          3.428000  ...          0.246000  0.000000
2                 6.888235          3.100000  ...          2.123529  2.000000
3                 5.939394          2.754545  ...          1.445455  1.242424
'''



































