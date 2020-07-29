# -*- coding: utf-8 -*-
"""
교차검정(CrossValidation)
"""

from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report



# 1. dataset load
digit = load_digits()
X = digit.data
y = digit.target

X.shape  # (1797, 64)
y  # array([0, 1, 2, ..., 8, 9, 8])
y.shape  # (1797,)


# 2. model 생성
rf = RandomForestClassifier()
rfmodel = rf.fit(X, y)

pred = rfmodel.predict(X)  # class 예측치
pred  # array([0, 1, 2, ..., 8, 9, 8])
pred2 = rfmodel.predict_proba(X)  # 확률로 예측
pred2
'''  중첩리스트
        0     1     2     ...  7     8     9 일 확률
        
array([[1.  , 0.  , 0.  , ..., 0.  , 0.  , 0.  ],
       [0.  , 0.98, 0.01, ..., 0.  , 0.  , 0.  ],
       [0.  , 0.08, 0.87, ..., 0.  , 0.04, 0.  ],
       ...,
       [0.01, 0.04, 0.02, ..., 0.  , 0.82, 0.01],
       [0.02, 0.  , 0.01, ..., 0.  , 0.02, 0.91],
       [0.  , 0.  , 0.  , ..., 0.  , 0.95, 0.02]])
'''

# 확률 -> index(10진수)
pred2_dit = pred2.argmax(axis=1)
pred2_dit  #  array([0, 1, 2, ..., 8, 9, 8], dtype=int64) 
'''
numpy.argxxx()  : index 반환 함수
 - argsort()
 - argmax() 원소들 중 가장 값이 큰 것의 인덱스를 반환
 - argmin() 
'''


acc = accuracy_score(y, pred)
acc  # 1.0

acc2 = accuracy_score(y, pred2_dit)
acc2  #  1.0



# Croww Validation
score = cross_validate(rfmodel, X, y, scoring='accuracy', cv=5)
score
''' dict 형식
{'fit_time': array([0.21821547, 0.2186985 , 0.2186985 , 0.20419979, 0.2186985 ]),
 'score_time': array([0.        , 0.        , 0.01562166, 0.        , 0.01562119]),
 'test_score': array([0.925     , 0.91944444, 0.96100279, 0.96657382, 0.92479109])}  >> 이게 중요함
                                                                                  >> 이걸 산술평균해서 최종 분류정확도 결정
'''
test_score = score['test_score']
test_score.mean()  # 0.9393624264933458



























