'''
 문) load_breast_cancer 데이터 셋을 이용하여 다음과 같이 Decision Tree 모델을 생성하시오.
 <조건1> 75:25비율 train/test 데이터 셋 구성 
 <조건2> y변수 : cancer.target, x변수 : cancer.data 
 <조건3> 중요변수 확인 

'''
import pandas as pd
from sklearn import model_selection
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import export_text
from sklearn.metrics import accuracy_score, confusion_matrix

# 데이터 셋 load 
cancer = load_breast_cancer()
print(cancer)
print(cancer.DESCR)
cancer.target_names  # array(['malignant', 'benign'], dtype='<U9')
names = cancer.feature_names

# <조건2> y변수 : cancer.target, x변수 : cancer.data 
# 변수 선택 
X = cancer.data
y = cancer.target
X.shape  # (569, 30)
y.shape  # (569,)


# <조건1> 75:25비율 train/test 데이터 셋 구성 
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=123)
x_train.shape  # (426, 30)
x_test.shape  # (143, 30)
y_train.shape  # (426,)



# <조건3> 중요변수 확인
# 1) 지니 불순도
dtcg = DecisionTreeClassifier()
modelg = dtcg.fit(x_train, y_train)

tree.plot_tree(modelg)
print(export_text(modelg))
'''  max_depth=3일때 출력창
|--- feature_20 <= 16.80
|   |--- feature_27 <= 0.14
|   |   |--- feature_12 <= 6.60
|   |   |   |--- class: 1
|   |   |--- feature_12 >  6.60
|   |   |   |--- class: 0
|   |--- feature_27 >  0.14
|   |   |--- feature_21 <= 25.67
|   |   |   |--- class: 1
|   |   |--- feature_21 >  25.67
|   |   |   |--- class: 0
|--- feature_20 >  16.80
|   |--- feature_6 <= 0.07
|   |   |--- feature_1 <= 18.84
|   |   |   |--- class: 1
|   |   |--- feature_1 >  18.84
|   |   |   |--- class: 0
|   |--- feature_6 >  0.07
|   |   |--- feature_13 <= 16.36
|   |   |   |--- class: 1
|   |   |--- feature_13 >  16.36
|   |   |   |--- class: 0
'''
names[20]  # 'worst radius'
# [조건 3 정답] worst radius가 지니계수에 의해서 가장 중요한 변수로 확인됨.

# 1) 엔트로피
dtce = DecisionTreeClassifier(criterion='entropy')
modele = dtce.fit(x_train, y_train)

tree.plot_tree(modele)
print(export_text(modele))
'''  max_depth=3일때 출력창
|--- feature_20 <= 16.80
|   |--- feature_27 <= 0.14
|   |   |--- feature_1 <= 21.43
|   |   |   |--- class: 1
|   |   |--- feature_1 >  21.43
|   |   |   |--- class: 1
|   |--- feature_27 >  0.14
|   |   |--- feature_21 <= 26.90
|   |   |   |--- class: 1
|   |   |--- feature_21 >  26.90
|   |   |   |--- class: 0
|--- feature_20 >  16.80
|   |--- feature_6 <= 0.07
|   |   |--- feature_1 <= 18.84
|   |   |   |--- class: 1
|   |   |--- feature_1 >  18.84
|   |   |   |--- class: 0
|   |--- feature_6 >  0.07
|   |   |--- feature_21 <= 15.43
|   |   |   |--- class: 1
|   |   |--- feature_21 >  15.43
|   |   |   |--- class: 0
'''
names[20]  # 'worst radius'
# [조건 3 정답] worst radius가 엔트로피에 의해서도 가장 중요한 변수로 확인됨.













## (+) 모델 분류정확도
y_pred = modelg.predict(x_test)
y_true = y_test

acc = accuracy_score(y_true, y_pred)
acc  #  0.965034965034965  ** max_depth = 3 일 때는  0.9790209790209791

conmax = confusion_matrix(y_true, y_pred)
conmax
'''
array([[51,  3],
       [ 2, 87]], dtype=int64)
'''


y_pred2 = modele.predict(x_test)

acc2 = accuracy_score(y_true, y_pred2)
acc2  #  0.9790209790209791  ## 얘는 max_depth = 3 일 때와 같음

conmax = confusion_matrix(y_true, y_pred2)
conmax
'''
array([[52,  2],
       [ 1, 88]], dtype=int64)
'''