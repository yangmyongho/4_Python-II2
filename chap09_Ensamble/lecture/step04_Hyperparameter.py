# -*- coding: utf-8 -*-
"""
Random Forest Hyper parameter
    step02_RandomForest 참고
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score



# 1. dataset load
wine = load_wine()
wine.feature_names
'''
['alcohol',
 'malic_acid',
 'ash',
 'alcalinity_of_ash',
 'magnesium',
 'total_phenols',
 'flavanoids',
 'nonflavanoid_phenols',
 'proanthocyanins',
 'color_intensity',
 'hue',
 'od280/od315_of_diluted_wines',
 'proline']
'''
wine.target_names  # array(['class_0', 'class_1', 'class_2']  -> 분류분석에 적합

X = wine.data
y = wine.target
X.shape  # (178, 13)
y.shape  # (178,)


# 2. RF model
rf = RandomForestClassifier()
'''
[parameters]
- n_estimators : integer, optional (default=100)             >> 트리수
        The number of trees in the forest.
- criterion : string, optional (default="gini") or entropy   >> 중요변수 선정
- max_depth : integer or None, optional (default=None)       >> 트리의 깊이
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
- min_samples_split : int, float, optional (default=2)       >> 노드 분할 참여할 최소 샘플수
        The minimum number of samples required to split an internal node:
- min_samples_leaf : int, float, optional (default=1)        >> 단(터미널) 노드 분할 최소수
        The minimum number of samples required to be at a leaf node.
- max_features : int, float, string or None, optional (default="auto")  >> 최대 x 변수 사용수
        The number of features to consider when looking for the best split:
- random_state : int, RandomState instance or None, optional (default=None)
- n_jobs : int or None, optional (default=None)              >> cpu 수
        The number of jobs to run in parallel.
'''

model = rf.fit(X, y)



# 3. Grid Search
from sklearn.model_selection import GridSearchCV

# 1) params 설정
# grid : dict 형식 =  [{'object_C' : params_range}, {'object_gamma' : params_range}, ...]
params= {'n_estimators':[100,200,300,400], 
          'max_depth':[3,6,8,10], 
          'min_samples_split':[2,3,4,5],
          'min_samples_leaf':[1,3,5,7]}   # 리스트로 씌울 필요가 없음. 대괄호가 하나밖에 없으므로

# 2) GridSearch 객체 : 가장높은 score를 가진 hyper parameter 찾기
gs = GridSearchCV(estimator = model, param_grid=params, scoring='accuracy', cv=5, n_jobs=-1)
# cpu 수는 '내 컴퓨터'에 커서를 두고 마우스 오른쪽을 클릭하여 '장치 관리자'의 '프로세서'를 확인하면 알 수 있음
# 현재 이 컴퓨터의 컴퓨터 수는 12개이고,  n_jobs=-1 로 설정하면 해당 컴퓨터에서 사용할 수 있는 모든 cpu를 사용한다.
model = gs.fit(X, y) # Grid Search 형 모델

acc = model.score(X, y)
acc  # 1.0

model.best_params_ 
'''
{'max_depth': 6,
 'min_samples_leaf': 3,
 'min_samples_split': 5,
 'n_estimators': 100}
'''





























































