# -*- coding: utf-8 -*-
"""
step03_SMS_spam_classification.py

NB vs SVM : 희소행렬(고차원)
 - 가중치 적용 : Tfidf (전체 빈도수에 역수를 취해서)
 
 >> NB는 통계적 분류기이기 때문에 간단한 통계 계산으로 확률을 예측하므로 비교적 빠름
 >> SVM은 Hyper parameter를 조절해 높은 분류정확도를 얻을 수 있다.
"""

from sklearn.naive_bayes import MultinomialNB  # nb model
from sklearn.svm import SVC  # svm model
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np  # npy file load  # chap07에서 만든 data/spam_data.npy 


# 1. dataset load
x_train, x_test, y_train, y_test = np.load("C:\\ITWILL\\Work\\4_Python-II\\workspace\\chap07_Crawling\\data/spam_data.npy",
                                           allow_pickle=True)
x_train.shape  # (3901, 4000)
x_test.shape  # (1673, 4000)
len(y_train)  # 3901
len(y_test)  # 1673
# list -> numpy
y_train = np.array(y_train)
y_test = np.array(y_test)
y_train.shape  # (3901,)
y_test.shape  # (1673,)  >> 같은 일차원이지만 리스트는 호출할 수 있는 멤버나 연산에 제한이 있다.


# 2. NB model
nb = MultinomialNB()
model = nb.fit(X = x_train, y = y_train)

y_pred = model.predict(x_test)
y_true = y_test

# model evaluation
acc = accuracy_score(y_true, y_pred)
acc  # 0.9802749551703527

conmat = confusion_matrix(y_true, y_pred)
conmat
'''
array([[1435,    0],
       [  33,  205]], dtype=int64)
>> 햄, 스팸(0햄, 1스팸) 더미변수화 되어있음'''

acc2 = (conmat[0,0] + conmat[1,1]) / conmat.sum()
acc2  # 0.9802749551703527

# 스팸메일에 대한 분류정확도
197 / (44+197)  # 0.8174273858921162


# 3. SVM model : 속도 조금 더 느림
svc = SVC(gamma='auto')
model_svc = svc.fit(X = x_train, y = y_train)

y_pred2 = model_svc.predict(x_test)
y_true2 = y_test

# model evaluation
acc = accuracy_score(y_true2, y_pred2)
acc  # 0.8577405857740585

conmat = confusion_matrix(y_true2, y_pred2)
conmat
'''
array([[1435,    0],
       [ 238,    0]], dtype=int64)
>> 걍 죄다 햄메일로 분류했음'''
acc2 = 1435/(1435+238)
acc2  # 0.8577405857740585

### kernel = 'linear'
svc = SVC(kernel = 'linear')
model_svc = svc.fit(X = x_train, y = y_train)

y_pred2 = model_svc.predict(x_test)
y_true2 = y_test

# model evaluation
acc = accuracy_score(y_true2, y_pred2)
acc  # 0.9838613269575612

conmat = confusion_matrix(y_true2, y_pred2)
conmat
'''
array([[1432,    3],
       [  24,  214]], dtype=int64)
'''
print('스팸성 분류율 :', 214/(24+214))  # 스팸성 분류율 : 0.8991596638655462




