# -*- coding: utf-8 -*-
"""
C:\ITWILL\Work\4_Python-II\workspace\chap08_Classification\lecture
"""

# from module import function
import numpy as np
from step01_kNN_data import data_set

# dataset 생성
know, not_know, cate = data_set()
know.shape  # (4, 2) # 알려진 집단
not_know # array([1.6 , 0.85]) # 알려지지 않은 집단
cate # array(['A', 'A', 'B', 'B'], dtype='<U1')

# 거리계산식 : 차 > 제곱 > 합 > 제곱근
diff = know -not_know
diff = know - not_know
diff
'''
- 넘파이 구조이기 때문에 행렬과 벡터의 연산이 가능하다
array([[-0.4 ,  0.25],
       [-0.6 ,  0.15],
       [ 0.2 , -0.05],
       [ 0.4 ,  0.05]])
'''
square_diff = diff ** 2
square_diff
'''
array([[0.16  , 0.0625],
       [0.36  , 0.0225],
       [0.04  , 0.0025],
       [0.16  , 0.0025]])
'''
sum_square_diff = square_diff.sum(axis = 1) # 행단위 합계
sum_square_diff # array([0.2225, 0.3825, 0.0425, 0.1625])
sum_square_diff = square_diff.sum(axis=1)
sum_square_diff

distance = np.sqrt(sum_square_diff)
distance = np.sqrt(sum_square_diff)
distance #  array([0.47169906, 0.61846584, 0.20615528, 0.40311289])
cate # ['A', 'A', 'B', 'B']
# 3번째 - 4번째 - 1번째 순으로 거리가 가깝다.
# 거리가 가까운 k개(3개) 선정하고 그것의 레이블을 봤을 때 b가 2개, a가 1개이므로 b그룹에 속한다.
# 오름차순 - 인덱스 반환하도록 해보자.

sortDist = distance.argsort() # 넘파이에서 지원하는 오름차순 정렬 후 인덱스 정보 반환하는 argsort 함수
sortDist = distance.argsort()
sortDist # array([2, 3, 0, 1], dtype=int64)
sortDist[0]
cate[sortDist[0]]
result = cate[sortDist]
result # array(['B', 'B', 'A', 'A'], dtype='<U1') # 0 1 2 3에 해당하는 카테를 알려줌

# k = 3 - 최근접 이웃 3개
k3 = result[:3] # array(['B', 'B', 'A'], dtype='<U1')
k3 = result[:3]

# dict
classify_re = {}
for key in k3 :
    classify_re[key] = classify_re.get(key,0) +1

classify_re # {'B': 2, 'A': 1}

vote_re = max(classify_re)
vote_re = max(classify_re)
vote_re # 'B'
print('분류결과 :', vote_re) # 분류결과 : B


# 알고리즘 함수로 정의해보자.
def knn_classfy(know, not_know, cate, k):
    # 거리계산식
    diff = know -not_know
    diff
    square_diff = diff ** 2
    square_diff
    sum_square_diff = square_diff.sum(axis = 1) # 행단위 합계
    sum_square_diff 
    distance = np.sqrt(sum_square_diff)
    
    # 단계2 : 오름차순 정렬 -> index
    sortDist = distance.argsort()  # 내용을가지고 정렬
    
    # 단계3 : 최근접 이웃(k=3)
    class_result = {} # 빈 set
    
    for i in range(k): # k=3(0~2)
        key = cate[sortDist[i]]
        class_result[key] = class_result.get(key, 0) + 1
        
    return class_result

def knn_classify(know, not_know, cate, k):
    diff = know - not_know
    square_diff = diff**2
    sum_square_diff = square_diff.sum(axis=1)
    distance = np.sqrt(sum_square_diff)
    
    sortDist = distance.argsort()
    
    for i in range(k) :
        key = cate[sortDist[i]]
        class_result[key] = class_result.get(key,0) + 1
    
    return class_result



# 적용해보자.
class_result = knn_classfy(know, not_know, cate,3) # {'B': 2, 'A': 1}
class_result = knn_classify(know, not_know, cate, 3)
print('분류결과 :', max(class_result, key=class_result.get)) # 분류결과 : B

    
    















