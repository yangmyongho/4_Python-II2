# -*- coding: utf-8 -*-
"""
Decision Tree 모델
 - 중요변수 선정 기준 : GINI, Entropy
 - GINI : 불확실성을 개선하는데 기여하는 척도
 - Entropy : 불확실성을 나타내는 척도
 - both 작을수록 불확실성도 낮다.
"""

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_wine
from sklearn.tree import DecisionTreeClassifier, export_graphviz  # tree model, 파일로 시각화, gwedit을 실행해서 확인.
from sklearn.metrics import accuracy_score, confusion_matrix
# tree 시각화 관련
from sklearn.tree.export import export_text  # tree 구조 텍스트
from sklearn import tree


iris = load_iris()
iris.target_names  # array(['setosa', 'versicolor', 'virginica'], dtype='<U10')
names = iris.feature_names
names
'''
['sepal length (cm)',
 'sepal width (cm)',
 'petal length (cm)',
 'petal width (cm)']
'''


X = iris.data
y = iris.target

X.shape  # (150, 4)
y.shape  # (150,)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

help(DecisionTreeClassifier)

# model 생성
dtc = DecisionTreeClassifier(criterion='gini', random_state=123, max_depth=3)
model = dtc.fit(X=x_train, y=y_train)

dtc2 = DecisionTreeClassifier(criterion='entropy', random_state=123)  # depth 자동 결정
model2 = dtc2.fit(X=x_train, y=y_train)

# tree 모델 시각화
tree.plot_tree(model)

print(export_text(model))
'''
|--- feature_2 <= 2.60  >> feature_names를 직접 받을 수도 있음 : 3번 칼럼 분류조건(왼쪽노드)
|   |--- class: 0       >> ' setosa' 100% 분류
|--- feature_2 >  2.60  >> 3번칼럼 분류조건(오른쪽노드)
|   |--- feature_3 <= 1.65
|   |   |--- feature_2 <= 5.00
|   |   |   |--- class: 1
|   |   |--- feature_2 >  5.00
|   |   |   |--- class: 2
|   |--- feature_3 >  1.65
|   |   |--- feature_2 <= 4.85
|   |   |   |--- class: 2
|   |   |--- feature_2 >  4.85
|   |   |   |--- class: 2
'''
print(export_text(model, names))  # names = iris.feature_names
'''
|--- petal length (cm) <= 2.60
|   |--- class: 0
|--- petal length (cm) >  2.60
|   |--- petal width (cm) <= 1.65
|   |   |--- petal length (cm) <= 5.00
|   |   |   |--- class: 1
|   |   |--- petal length (cm) >  5.00
|   |   |   |--- class: 2
|   |--- petal width (cm) >  1.65
|   |   |--- petal length (cm) <= 4.85
|   |   |   |--- class: 2
|   |   |--- petal length (cm) >  4.85
|   |   |   |--- class: 2
'''

tree.plot_tree(model2)  # 6개의 depth
print(export_text(model2, names))



# model evaluation
y_pred = model.predict(x_test)
y_true = y_test

acc = accuracy_score(y_true, y_pred)
acc  # 0.9555555555555556

conmat = confusion_matrix(y_true, y_pred)
conmat
'''
array([[14,  0,  0],                 >> 100% 분류 'setosa'
       [ 0, 17,  1],
       [ 0,  1, 12]], dtype=int64)
'''



y_pred2 = model2.predict(x_test)
y_true2 = y_test

acc = accuracy_score(y_true2, y_pred2)
acc  # 0.9555555555555556  : 근데 분류정확도는 똑같음  >> 원래는 depth가 높아지면 분류정확도는 높아져야만 함. but, 동시에 과적합도 생긴다.

conmat = confusion_matrix(y_true2, y_pred2)
conmat
'''
array([[14,  0,  0],
       [ 0, 17,  1],
       [ 0,  1, 12]],
'''



###############################################
############ model overfitting
###############################################

wine = load_wine()
X = wine.data
y = wine.target
x_name = wine.feature_names  # x 변수명

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

dt = DecisionTreeClassifier()  # default
model = dt.fit(x_train, y_train)

train_score = model.score(x_train, y_train)
test_score = model.score(x_test, y_test)
train_score  # 1.0
test_score  # 0.9444444444444444

tree.plot_tree(model)  # max_depth=6

# max_depth=3
dt2 = DecisionTreeClassifier(max_depth=3) 
model2 = dt2.fit(x_train, y_train)

train_score = model2.score(x_train, y_train)
test_score = model2.score(x_test, y_test)
train_score  # 0.9838709677419355
test_score  # 0.9074074074074074  >> 정확도가 조금 떨어졌지만 train/test간 차이가 줄어듬. overfitting 줄어듬

tree.plot_tree(model2)


export_graphviz(model,
                out_file='C:\\ITWILL\\Work\\4_Python-II\\workspace\\chap08_Classification\\lecture02_NB_model\\DT_tree_graph.dot',
                feature_names=x_name, max_depth=3, class_names=True)
##  >> gvedit.exe 를 통해 그림으로 볼 수 있음.




































