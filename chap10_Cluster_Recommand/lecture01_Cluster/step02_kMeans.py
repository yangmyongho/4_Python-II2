# -*- coding: utf-8 -*-
"""
kMeans 알고리즘
 - 비계층적(확인적) 군집분석
 - 군집수(k) 알고 있는 경우 이용
"""

import pandas as pd
import numpy as np  # array
#from sklearn.cluster import kMeans  # model
from sklearn.cluster import KMeans  # model
import matplotlib.pyplot as plt


# 1. txt dataset load as np.array

# text file -> numpy
def dataMat(file) :
    dataset = []  # data mat ? 저장
    
    f = open(file, mode='r')  # file object
    lines = f.readlines()  # 1.658985	4.285136
    for line in lines:
        cols = line.split('\t')
        
        rows = []  # x, y 벡터형태로 저장
        for col in cols:  # '1.658985'
            rows.append(float(col))  # [1.658985, 4.285136]
        
        dataset.append(rows)  # [[rows], [rows], ..., [rows]]
        
    return np.array(dataset)


dataset = dataMat("C:\\ITWILL\\Work\\4_Python-II\\data/testSet.txt")


# numpy -> pandas
dataset_df = pd.DataFrame(dataset, columns=['x', 'y'])



# 2. kMeans model : k=4
k = KMeans(n_clusters=4, algorithm='auto')
k
'''
KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
       n_clusters=4, n_init=10, n_jobs=None, precompute_distances='auto',
       random_state=None, tol=0.0001, verbose=0)
'''

model = k.fit(dataset_df)
pred = model.predict(dataset_df)
pred  # 0 ~ 3


# 각 cluster의 center
centers = model.cluster_centers_
'''
array([[-3.38237045, -2.9473363 ,  0.        ],  각 3번째 칼럼에 나온 숫자번째 클러스터의 센터값들
       [ 2.6265299 ,  3.10868015,  3.        ],
       [-2.46154315,  2.78737555,  1.        ],
       [ 2.80293085, -2.7315146 ,  2.        ]])
'''




# 3. 시각화
# 군집 형성 현황
dataset_df['cluster'] = pred
dataset_df.info()
''' 0   x        80 non-null     float64
 1   y        80 non-null     float64
 2   cluster  80 non-null     int32 
'''
dataset_df.head()
'''
          x         y  cluster
0  1.658985  4.285136        3
1 -3.453687  3.424321        1
2  4.838138 -1.151539        2
3 -5.379713 -3.362104        0
4  0.972564  2.924086        3
'''

plt.scatter(x=dataset_df['x'], y=dataset_df['y'], c=dataset_df['cluster'], marker='o')


# 중심점
plt.scatter(x=centers[:,0], y=centers[:,1], c='red', marker='D', s=100)



grp = dataset_df.groupby('cluster')
grp.mean()
'''
                x         y
cluster                    
0       -3.382370 -2.947336
1       -2.461543  2.787376
2        2.802931 -2.731515
3        2.626530  3.108680
'''












