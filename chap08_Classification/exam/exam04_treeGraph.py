# -*- coding: utf-8 -*-
"""
문) tree_data.csv 파일의 변수를 이용하여 아래 조건으로 DecisionTree model를 생성하고,
    의사결정 tree 그래프를 시각화하시오.
    
 <변수 선택>   
 x변수 : iq수치, 나이, 수입, 사업가유무, 학위유무
 y변수 : 흡연유무
 
 <그래프 저장 파일명> : smoking_tree_graph.dot
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree

tree_data = pd.read_csv("C:\\ITWILL\\Work\\4_Python-II\\data/tree_data.csv")
tree_data.head()
print(tree_data.info())
'''
iq         6 non-null int64 - iq수치
age        6 non-null int64 - 나이
income     6 non-null int64 - 수입
owner      6 non-null int64 - 사업가 유무
unidegree  6 non-null int64 - 학위 유무
smoking    6 non-null int64 - 흡연 유무 : y변수
'''
cols = list(tree_data.columns)
cols

x_cols = cols[:5]
x_cols
y_col = cols[5]
y_col

X = tree_data[x_cols]
y = tree_data[y_col]


# gini
dtg = DecisionTreeClassifier()
modelg = dtg.fit(X, y)

tree.plot_tree(modelg, max_depth=None, feature_names=x_cols)
# 중요 변수는 income(수입)이다.
export_graphviz(modelg,
                out_file='C:\\ITWILL\\Work\\4_Python-II\\data/smoking_tree_graph.dot',
                max_depth=None, feature_names=x_cols)


# entropy
dte = DecisionTreeClassifier(criterion='entropy')
modele = dte.fit(X, y)

tree.plot_tree(modele, max_depth=None, feature_names=x_cols)
export_graphviz(modele,
                out_file='C:\\ITWILL\\Work\\4_Python-II\\data/smoking_tree_graph_entropy.dot',
                max_depth=None, feature_names=x_cols)
