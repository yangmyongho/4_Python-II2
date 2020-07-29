# -*- coding: utf-8 -*-
"""
SVM model
 - 선형 SVM, 비선형 SVM
 - Hyper parameter(kernel, C(cost), gamma(비선형일때)) : 데이터의 특성에 따라 파라미터가 바뀌는.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score









# 1. dataset load
iris = pd.read_csv("C:\\ITWILL\\Work\\4_Python-II\\data/iris.csv")
iris.info()
"""
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   Sepal.Length  150 non-null    float64  >> 연속형(x)
 1   Sepal.Width   150 non-null    float64  >> 연속형(x)
 2   Petal.Length  150 non-null    float64  >> 연속형(x)
 3   Petal.Width   150 non-null    float64  >> 연속형(x)
 4   Species       150 non-null    object   >> 범주형(y)
"""


# 2. x,y 변수 선택
cols = list(iris.columns)
cols

x_cols = cols[:4]  # ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']
y_col = cols[-1]  #  'Species'


# 3. train(60%) / test(40%) dataset slpit
train, test = train_test_split(iris, test_size = 0.4, random_state = 123)
train.shape  # (90, 5)
test.shape  # (60, 5)


# 4. SVM model 생성
svc = SVC(C=1.0, gamma='auto', kernel='rbf')  # default : (비선형 svm모델)
model = svc.fit(X = train[x_cols], y=train[y_col])
model  # model information

svc2 = SVC(c=1.0, kernel='linear')
model2 = svc.fit(X = train[x_cols], y=train[y_col])

y_pred = model.predict(test[x_cols])
y_true = test[y_col]
y_pred2 = model2.predict(test[x_cols])
y_true2 = test[y_col]


# 5. model 평가
# 비선형
acc = accuracy_score(y_true, y_pred)
acc  # 0.9666666666666667

# 선형
acc2 = accuracy_score(y_true2, y_pred2)
acc2  # 0.9666666666666667 >> 선형이나 비션형이나 정확도가 같음



#################################################
### Grid Search
#################################################
# Hyper parameter(kernel, C, gamma) 최적의 파라미터 찾기

# Cost(C), gamma
params = [0.001, 0.01, 0.1, 1, 10, 100]  # 10^-3 ~ 10^2
kernel = ['linear', 'rbf']  # kernel parameters
best_score = 0  # best score 초기값
best_parameter = {}  # dict : ex : {'kernel' : 'linear', 'C' : 1}

for k in kernel :
    for gamma in params :
        for C in params :
            svc = SVC(kernel = k, gamma = gamma, C = C)
            model = svc.fit(X=train[x_cols], y=train[y_col])
            score = model.score(test[x_cols], test[y_col])
            
            if score > best_score :
                best_score = score
                best_parameter =  {'kernel' : k, 'gamma' : gamma, 'C' : C}
                
print('best score =', best_score)  #best score = 1.0
print('best Hyper parameters =', best_parameter)  # best Hyper parameters = {'kernel': 'rbf', 'gamma': 0.01, 'C': 100}
















