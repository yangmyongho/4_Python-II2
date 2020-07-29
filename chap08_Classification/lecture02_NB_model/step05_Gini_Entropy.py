# -*- coding: utf-8 -*-
"""
Gini 불순도(Inpurity), Entropy
 - tree model 에서 중요변수 선정 기준
 - 정보이득 = base 지수 - Gini 불순도 or Entropy
 - 정보이득이 클 수록 중요변수로 본다.
 - Gini impurity = sum(p * (1-p))
 - Entropy = -sum(p * log(p)
"""

import numpy as np

# 1. 불확실성 큰 경우
x1 = 0.5; x2 = 0.5

gini = sum([x1 * (1-x1), x2 * (1-x2)])
print('gini =', gini)  # gini = 0.5

entropy = -sum([x1*np.log2(x1),x2*np.log2(x2)])
print('entropy =', entropy)  # entropy = 1.0  : 엔트로피는 1일 때 불확실성이 최대.
 
'''
cost(loss) function : 정답과 예측치 -> 오차 반환 함수
x1 -> y_true, x2 -> y_pred
y_true = x1*np.log2(x1)
y_pred = x2*np.log2(x2)
'''

y_true = x1*np.log2(x1)
y_pred = x2*np.log2(x2)
cost = -sum([y_true, y_pred])
cost  # 비용(손실)이 클수록 불확실성이 큼.


# 2. 불확실성이 작은 경우
x1 = 0.9; x2 = 0.1
gini = sum([x1 * (1-x1), x2 * (1-x2)])
print('gini =', gini)  # gini = 0.18  줄었다.

entropy = -sum([x1*np.log2(x1),x2*np.log2(x2)])
print('entropy =', entropy)  # 0.4689955935892812 얘도 줄었다. 불확실성이 작아졌다.

y_true = x1*np.log2(x1)
y_pred = x2*np.log2(x2)
cost = -sum([y_true, y_pred])
cost # 0.4689955935892812


############################################
#### dataset 적용
############################################

def createDataSet(): 
    dataSet = [[1, 1, 'yes'], 
               [1, 1, 'yes'], 
               [1, 0, 'no'], 
               [0, 1, 'no'],   # 각각 x1, x2, y
               [0, 1, 'no']]   # 5행 3열 array
    columns = ['dark_clouds','gust'] # X1,X2,label
    return dataSet, columns

dataSet, columns = createDataSet()

dataSet
type(dataSet)  # list
columns
type(columns)  # list
# list -> np (선형대수 연산 안되므로)
dataSet = np.array(dataSet)
columns = np.array(columns)
dataSet.shape  # (5, 3)
columns.shape  # (2,)

X = dataSet[:, :2]
X
'''     x1    x2
array([['1', '1'],
       ['1', '1'],
       ['1', '0'],
       ['0', '1'],
       ['0', '1']]
'''
y = dataSet[:,2]
y  # array(['yes', 'yes', 'no', 'no', 'no']

# dummy( y전처리)
y = [1 if d == 'yes' else 0 for d in y]
y  # [1, 1, 0, 0, 0]


from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
from sklearn import tree

dt = DecisionTreeClassifier()
model = dt.fit(X, y)

y_pred = model.predict(X)

acc = accuracy_score(y, y_pred)
acc  # 1.0

# 중요변수
tree.plot_tree(model)
columns[0]  # 'dark_clouds'  : 중요변수

export_graphviz(model,
                out_file='C:\\ITWILL\\Work\\4_Python-II\\workspace\\chap08_Classification\\lecture02_NB_model/dataset_tree.dot',
                max_depth=3, feature_names = columns)


# entropy
dt = DecisionTreeClassifier(criterion='entropy')
model = dt.fit(X, y)
tree.plot_tree(model)
columns[0]  # 'dark_clouds'  : 중요변수

export_graphviz(model,
                out_file='C:\\ITWILL\\Work\\4_Python-II\\workspace\\chap08_Classification\\lecture02_NB_model/dataset_tree_entropy.dot',
                max_depth=3, feature_names = columns)
























