# -*- coding: utf-8 -*-
"""
NB 모델
 GaussianNB : x변수가 실수형(연속형)이고, 정규분포 형태를 띠고 있을 때
 MultinomialNB : 희소행렬과 같은 고차원 데이터를 이용하여 다항분류 할 때
"""

import pandas as pd  # csv file read
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB  # model class
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score  # model evaluation
from scipy import stats  # 정규분포 검정


################################
### GaussianNB 
################################

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

# 정규성 검정
stats.shapiro(iris['Sepal.Width'])  # (0.9849170446395874, 0.10113201290369034)
                                    # p 0.05 이상이므로 "정규분포와 차이가 없다."
stats.shapiro(iris['Sepal.Length'])  # 얘 빼고 다 정규분포
stats.shapiro(iris['Petal.Length'])
stats.shapiro(iris['Petal.Width'])


# 2. x,y 변수 선택
cols = list(iris.columns)
cols

x_cols = cols[:4]  # ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']
y_col = cols[-1]  #  'Species'


# 3. train / test dataset slpit
train, test = train_test_split(iris, test_size = 0.3, random_state = 123)
train.shape  # (105, 5)
test.shape  # (45, 5)


# 4. model - using class GausianNB
nb = GaussianNB()
model = nb.fit(X=train[x_cols], y=train[y_col])
model  # GaussianNB(priors=None, var_smoothing=1e-09)  : 어떤 parameter로 모델이 만들어졌는지, model에 대한 객체 정보를 알 수 있다.


# 5. model evaluation
y_pred = model.predict(test[x_cols])
y_true = test[y_col]

# 1) accuracy score  분류정확도
acc = accuracy_score(y_true, y_pred)
acc  # 0.9555555555555556

# 2) confusion matrix  교차분할표
con_mat = confusion_matrix(y_true, y_pred)
con_mat
"""
array([[18,  0,  0],
       [ 0, 10,  0],
       [ 0,  2, 15]], dtype=int64)
"""
acc2 = (con_mat[0,0] + con_mat[1,1] + con_mat[2,2] ) / con_mat.sum()
acc2  #  0.9555555555555556

# 3) f1 score  : y 불균형인 경우 : 정확률, 재현율 -> 조화평균
#acc3 = f1_score(y_true, y_pred)  # ValueError: Target is multiclass but average='binary'. Please choose another average setting, one of [None, 'micro', 'macro', 'weighted'].
acc3 = f1_score(y_true, y_pred, average='micro')
acc3  # 0.9555555555555556




################################
### MultinomialNB(ex.희소행렬)
################################
# chap06_regression/sklearn_category_data.py 참조

# 1. dataset load
from sklearn.datasets import fetch_20newsgroups
newsgroups = fetch_20newsgroups(subset='all') # 'train', 'test'
# Downloading 20news dataset.
newsgroups.target  # array([10,  3, 17, ...,  3,  1,  7])

newsgroups.target.shape  # (18846,)
newsgroups.filenames.shape  # (18846,)
newsgroups.data[0]
len(newsgroups.data)  # 18846
#newsgroups.feature_names  # AttributeError:
#newsgroups.columns


print(newsgroups.DESCR)



'''
x변수 : news 기사 내용(text 자료)
y변수 : 해당 news의 group(20개)
'''

cats = newsgroups.target_names[:4]  # y
'''
['alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware']
'''




# 2. text -> sparse matrix : fetch_20newsgroups(subset='train')
from sklearn.feature_extraction.text import TfidfVectorizer

news_train = fetch_20newsgroups(subset='train', categories=cats)
news_train.data  # text(x변수로 사용할 것.)

news_train.target  #  array([3, 2, 2, ..., 0, 1, 1], dtype=int64) : y변수

tfidf = TfidfVectorizer()
sparsemat = tfidf.fit_transform(news_train.data)
sparsemat
'''<2245x62227 sparse matrix of type '<class 'numpy.float64'>'
	with 339686 stored elements in Compressed Sparse Row format>'''
sparsemat.shape  # (2245, 62227)
print(sparsemat)



# 3. model
nb = MultinomialNB()
model = nb.fit(X=sparsemat, y=news_train.target)




# 4. model 평가 : fetch_20newsgroups(subset='test')
news_test = fetch_20newsgroups(subset='test', categories=cats)
news_test.data  # text
news_test.target  # array([1, 1, 1, ..., 1, 3, 3], dtype=int64)

# ** test 평가용 데이터셋을 만들 때는 fit_transform이 아닌 transform 사용**
sparse_test = tfidf.transform(news_test.data)
sparse_test.shape  # (1494, 62227)  : 단어 개수가 train set와 같냉.

'''
obj.fit_transform(train_data)
obj.transform(text_data)
'''


y_pred = model.predict(sparse_test)
y_true = news_test.target


acc = accuracy_score(y_true, y_pred)
acc  # 0.8520749665327979

conmat = confusion_matrix(y_true, y_pred)
conmat
'''
array([[312,   2,   1,   4],
       [ 12, 319,  22,  36],
       [ 16,  26, 277,  75],
       [  1,   8,  18, 365]], dtype=int64)'''
acc2 = (conmat[0,0] + conmat[1,1] + conmat[2,2] + conmat[3,3]) / conmat.sum()
acc2  # 0.8520749665327979

acc3 = f1_score(y_true, y_pred, average='micro')
acc3  # 0.8520749665327979

acc4 = f1_score(y_true, y_pred, average='macro')
acc4  # 0.8545568195295443

acc5 = f1_score(y_true, y_pred, average=None)
acc5  # Out[282]: array([0.94545455, 0.85752688, 0.77808989, 0.83715596])

acc6 = f1_score(y_true, y_pred, average='weighted')
acc6  # 0.8500070350296273




y_pred[:20]  # [1, 2, 1, 2, 3, 1, 3, 1, 0, 3, 1, 3, 0, 2, 3, 1, 0, 3, 0, 2]
y_true[:20]  # [1, 1, 1, 2, 3, 1, 3, 1, 0, 3, 3, 3, 0, 2, 3, 1, 0, 3, 2, 2]
             #     v                          v                       v





