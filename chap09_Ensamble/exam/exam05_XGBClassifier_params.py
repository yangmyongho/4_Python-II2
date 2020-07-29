# -*- coding: utf-8 -*-
"""
문) wine dataset을 이용하여 다음과 같이 다항분류 모델을 생성하시오. 
 <조건1> tree model 200개 학습
 <조건2> tree model 학습과정에서 조기 종료 100회 지정
 <조건3> model의 분류정확도와 리포트 출력   
"""
from xgboost import XGBClassifier # model
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine # 다항분류
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV


#################################
## 1. XGBoost Hyper Parameter
#################################

# 1. dataset load
X, y = load_wine(return_X_y = True)
X.shape  # (178, 13)
y  # 0 ~ 2



# 2. train/test 생성 
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
x_train.shape  # (124, 13)
x_test.shape  # (54, 13)
y_train
y_test




# 3. model 객체 생성
help(XGBClassifier)
xgb = XGBClassifier(n_estimators=200)
xgb


obj = XGBClassifier(colsample_bytree=1,
                    learning_rate=0.1, max_depth=3, 
                    min_child_weight=1,
                    n_estimators=200, 
                    objective="multi:softprob",
                    num_class=3)




# (+) parameter grid
params = {'colsample_bytree':[0.5, 0.7, 0.9],
          'learning_rate':[0.01, 0.1, 0.3],
          'max_depth':[3,5,6],
          'min_child_weight':[1,3],
          'n_estimators':[100,200]}

gs = GridSearchCV(estimator=xgb, param_grid=params, cv=5, verbose=True)



# 4. model 학습 조기종료 
evals = [(x_test), (y_test)]

model_early = obj.fit(x_train, y_train, eval_set = evals, early_stopping_rounds=200,
                     eval_metric='merror', verbose=True)

model_early = gs.fit(x_train, y_train, eval_metric='merror',
                early_stopping_rounds=100, eval_set=evals, verbose=True)
model_early  # objective='multi:softprob'



# 5. model 평가 
y_pred = model_early.predict(x_test)

model_early.best_params_
'''

{'colsample_bytree': 0.5,
 'learning_rate': 0.3,
 'max_depth': 3,
 'min_child_weight': 3,
 'n_estimators': 100}
'''
model_early.best_score_  # 0.9753333333333334

































