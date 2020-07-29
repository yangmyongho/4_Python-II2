# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Random Forest Ensemble model
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report  



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
rf

# train/test split
import numpy as np
idx = np.random.choice(a=X.shape[0], size=int(X.shape[0]*0.7), replace=False)  # X.shape[0]=178
x_train = X[idx]  # X[idx, :] 
len(idx)  # 124
y_train = y[idx]


idx_test = [ i for i in range(len(y)) if not i in idx ]
len(idx_test)  #54

x_test = X[idx_test]
y_test = y[idx_test]
len(y_test)  # 54

# model
model = rf.fit(X=x_train, y=y_train)

#predict
y_pred = model.predict(x_test)
y_true = y_test

# evaluating
con_mat = confusion_matrix(y_true, y_pred)
con_mat
'''
array([[19,  0,  0],
       [ 0, 19,  0],
       [ 0,  0, 16]], dtype=int64)
'''
acc = accuracy_score(y_true, y_pred)
acc  # 1.0

report = classification_report(y_true, y_pred)
print(report)
'''
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        19
           1       1.00      1.00      1.00        19
           2       1.00      1.00      1.00        16

    accuracy                           1.00        54
   macro avg       1.00      1.00      1.00        54
weighted avg       1.00      1.00      1.00        54
'''


# 중요변수
print('중요변수 :', model.feature_importances_)
'''
[0.1355671  0.02647958 0.01639293 0.01835803 0.03574079 0.0440059
 0.14252057 0.01571546 0.03085473 0.16151489 0.07932905 0.13019286
 0.16332811]'''  # >> 13개 list : gini계수를 근거로 얻어진 점수


# 중요변수 시각화
import matplotlib.pyplot as plt

x_size = X.shape[1]
plt.barh(range(x_size), model.feature_importances_)  # (y, x))

plt.yticks(range(x_size), wine.feature_names)
plt.xlabel('importance')
plt.show













