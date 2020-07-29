# -*- coding: utf-8 -*-
"""
XGBoost model : 분류트리(XGBClassifier)

> pip install xgboost
"""


# import test
from xgboost import XGBClassifier, XGBRegressor  # model
from xgboost import plot_importance  # x변수에 대한 중요변수 시각화
from sklearn.datasets import make_blobs  # 클러스터 데이터셋 생성
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt



# 1. dataset load
X, y = make_blobs(n_samples=2000, n_features=4, centers=2, cluster_std=2.5)
'''
n_samples : 데이터셋 크기
n_features : X변수
centers : Y변수 범주(클러스터) 수
cluster_std : 클러스터 표준편차(클수록 오분류 커짐, 그룹 사이 중첩 커짐), 1이 기본
'''

X.shape  # (2000, 4)
y.shape  # (2000,)
y  #array([1, 1, 0, ..., 1, 0, 0])

plt.scatter(x = X[:,0], y = X[:,1], s=100, c=y, marker='o')  # s size, c color marker 산점도 모양
plt.show()


# 2. train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# 3. model  생성
xgb = XGBClassifier()
help(XGBClassifier)
model = xgb.fit(x_train, y_train)
model  #  objective='binary:logistic'
'''
XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.300000012, max_delta_step=0, max_depth=6,
              min_child_weight=1, missing=nan, monotone_constraints='()',
              n_estimators=100, n_jobs=0, num_parallel_tree=1,
              objective='binary:logistic', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)
'''
model.score(x_test, y_test)  # 1.0



# 4. model evaluation
y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)
acc  # 1.0

report = classification_report(y_test, y_pred)
print(report)
'''
              precision    recall  f1-score   support
              정확률        재현율
           0       1.00      1.00      1.00       299
           1       1.00      1.00      1.00       301

    accuracy                           1.00       600
   macro avg       1.00      1.00      1.00       600
weighted avg       1.00      1.00      1.00       600
'''



# XGBoost는 시각화도구를 따로 제공한다.
# 5. 중요변수 시각화
fscore = model.get_booster().get_fscore()
fscore #  {'f0': 23}


plot_importance(model)
plt.show()






####################################### 다항으로 다시!
# 1. dataset load
X, y = make_blobs(n_samples=2000, n_features=4, centers=3, cluster_std=2.5)
'''
n_samples : 데이터셋 크기
n_features : X변수
centers : Y변수 범주(클러스터) 수
cluster_std : 클러스터 표준편차(클수록 오분류 커짐, 그룹 사이 중첩 커짐)
'''

X.shape  # (2000, 4)
y.shape  # (2000,)
y  #  array([0, 1, 1, ..., 1, 2, 2])  : 0~2 3개의 도메인

plt.scatter(x = X[:,0], y = X[:,1], s=100, c=y, marker='o')  # s size, c color marker 산점도 모양
plt.show()


# 2. train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# 3. model  생성
xgb = XGBClassifier()
help(XGBClassifier)
model = xgb.fit(x_train, y_train)
model  #  objective='multi:softprob' : 다항분류로 자동 전환
'''
XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.300000012, max_delta_step=0, max_depth=6,
              min_child_weight=1, missing=nan, monotone_constraints='()',
              n_estimators=100, n_jobs=0, num_parallel_tree=1,
              objective='multi:softprob', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=None, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)
>> n_estimators=100, max_depth=6,learning_rate=0.300000012(학습율 : 1에 가까울수록 빠른속도로 학습)
'''
model.score(x_test, y_test)  # 0.9816666666666667



# 4. model evaluation
y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)
acc  # 0.9816666666666667

report = classification_report(y_test, y_pred)
print(report)
'''
              precision    recall  f1-score   support

           0       0.97      0.97      0.97       183
           1       0.97      0.98      0.97       216
           2       1.00      1.00      1.00       201

    accuracy                           0.98       600
   macro avg       0.98      0.98      0.98       600
weighted avg       0.98      0.98      0.98       6
'''



# XGBoost는 시각화도구를 따로 제공한다.
# 5. 중요변수 시각화
fscore = model.get_booster().get_fscore()
fscore #  {'f3': 624, 'f0': 516, 'f2': 423, 'f1': 409}


plot_importance(model)
plt.show()











