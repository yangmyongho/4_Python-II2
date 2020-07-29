# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
1. XGBoost Hyper Parameter
2. model 학습 조기종료 : early stopping rounds
3. Best Hyper Parameter : Grid Search
"""

from xgboost import XGBClassifier, XGBRegressor  # model
from xgboost import plot_importance  # x변수에 대한 중요변수 시각화
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


# 1. XGBoost Hyper Parameter

X, y = make_blobs(n_samples=2000, n_features=4, centers=3, cluster_std=2.5)

X.shape  #(2000, 4)
y  # array([2, 1, 2, ..., 2, 2, 2]) 0~2

# train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# create model
xgb = XGBClassifier(colsample_bylevel=1, learning_rate=0.3, max_depth=6,
                    min_child_weight=1, n_estimators=200)
model = xgb.fit(x_train, y_train)  # 1st ~ 100th
model
# objective='multi:softprob'
# colsample_bylevel=1  : 트리모델 생성시 훈련셋 샘플링비율 : 100% 
# learning_rate=0.3  : 학습율(0.01~0.1 정도가 일반적)숫자가 작을수록 정밀도 up, 속도 down
# max_depth=6 : 트리의 깊이. 과적합에 영향
# min_child_weight=1 : 자식노드를 분할을 결정하는 최소 가중치 합 : 입력한 수를 기준. 트리의 깊이에 영향을 미치는 요인. 과적합에 영향
# n_estimators=200





# 2. model 학습 조기종료 : early stopping rounds
eval_set = [(x_test, y_test)]

model_early = xgb.fit(X, y, eval_set = eval_set, eval_metric='merror',
                      early_stopping_rounds=100, verbose=True)
# X, y : 모델을 학습할 수 있는 훈련 셋.
# eval_set : 모델을 평가할 수 있는 셋
# eval_metric='error' : 평가방법은 오류율을 기준으로.(이항분류:error, 다항분류:merror, 회귀:rmse)
# early_stopping_rounds=50 : 최소 50회 정도 해보고 acc가 좋아지지 않으면 그냥 종료해라. 100개까지 하지 말고.
# verbose=True : 모델을 학습하고  평가하는 과정을 화면상에 출력하겠다.

'''
Stopping. Best iteration:
[14]    validation_0-merror:0.000000   >>>  14번만돌려도 이미 더이상은 모델 정확도가 개선되지 않는다는 듯
'''

score_early = model_early.score(x_test, y_test)
score_early  # 1.0





# 3. 3. Best Hyper Parameter : Grid Search
from sklearn.model_selection import GridSearchCV

# default model object
xgb = XGBClassifier()

params = {'colsample_bylevel':[0.7, 0.9],
          'learning_rate':[0.01, 0.1],
          'max_depth':[3,5],
          'min_child_weight':[1,3],
          'n_estimator':[100,200]}

gs = GridSearchCV(estimator=xgb, param_grid=params, cv=5)  # 학습할 수 있는 객체
model = gs.fit(x_train, y_train, eval_set=eval_set, eval_metric='merror',verbose=True)  
model

# best score
model.best_score_  # 0.9914285714285714

# best parameter
model.best_params_
'''
{'colsample_bylevel': 0.7,
 'learning_rate': 0.01,
 'max_depth': 3,
 'min_child_weight': 1,
 'n_estimator': 100}
'''





































