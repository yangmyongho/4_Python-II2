# -*- coding: utf-8 -*-
"""
Pipeline vs Grid Search
 1. SVM Model
 2. Pipeline: model workflow(dataset 전처리 -> model 생성 -> model test)
 3. Grid Search : model tuning
"""

from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC  # model class
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler  # scaling(0 ~ 1)
from sklearn.pipeline import Pipeline  # model workflow 생성
import numpy as np


# 1. SVM model

# 1) dataset load
X, y = load_breast_cancer(return_X_y=True)
type(X)
X.shape  # (569, 30)
y.shape  # (569,)
# 얘네 둘다 np

X.mean(axis=0)  # 각 열의 평균
# 1st : [1.41272917e+01 : 14.~
# 4th : 6.54889104e+02  : 654.~
X.min()  # 0.0
X.max()  # 4254.0  : we should better go through scaling process

# 2) X 변수 정규화  : 전처리
scaler = MinMaxScaler(feature_range=(0,1)).fit(X)  # 1) scaler 객체 : default parameter
X_nor = scaler.transform(X)  # 2) 정규화

type(X_nor)  # sklearn.preprocessing._data.MinMaxScaler
# MinMaxScaler -> numpy
X_nor.mean(axis=0)
# 1st : 0.33822196
# 4th : 0.21692009
X_nor.min()  # 0.0
X_nor.max()  # 1.0000000000000002

x_train, x_test, y_train, y_test = train_test_split(X_nor, y, test_size=0.3)


# 3) SVM model 생성
svc = SVC(gamma='auto')
model = svc.fit(x_train, y_train)


# 4) model evaluation
score = model.score(x_test, y_test)
score  # 0.9590643274853801



# 2. 파이프라인 : model workflow

# 1) pipeline step : [ (step1:scaler), (step2:model), ...]  : 이렇게 list에 스텝명과 클래스를 대입
#                        'obj' ,     class
pipe_svc = Pipeline([ ('scaler', MinMaxScaler()), ('svc', SVC(gamma='auto')) ])

# 2) pipeline model 생성
model = pipe_svc.fit(x_train, y_train)

# 3) pipeline model test
score = model.score(x_test, y_test)
score  #  0.9590643274853801  똑같넹



# 3. Grid Search : model tuning
# Pipeline -> Grid Search -> model tuning
from sklearn.model_selection import GridSearchCV

help(SVC)
# C=1.0, kernel='rbf', degree=3, gamma='auto'(보통 auto로 함)

# 1) params 설정
params = [0.001, 0.01, 0.1, 1.0, 10.0 ,100.0, 1000.0]

# dict 형식 =  [{'object_C' : params_range}, {'object_gamma' : params_range}, ...]
params_grid = [{'svc__C':params, 'svc__kernel' :['linear']},   # 선형
               {'svc__C' : params, 'svc__gamma' : params, 'svc__kernel' :['rbf']}]  # 비선형

# 2) GridSearch 객체 : 가장높은 score를 가진 hyper parameter 찾기
gs = GridSearchCV(estimator = pipe_svc, param_grid=params_grid, scoring='accuracy', cv=10)
# scoring : 평가방법, cv : 교차검정
model = gs.fit(x_train, y_train) # Grid Search 형 모델

acc = model.score(x_test, y_test)
acc  # 0.9707602339181286

model.best_params_  #  {'svc__C': 10.0, 'svc__gamma': 0.1, 'svc__kernel': 'rbf'}fgfgfgdf



















