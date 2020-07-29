# -*- coding: utf-8 -*-
"""
step03_TfidifSparseMatrix2

1. csv file read : temp_spam_data2.csv
2. texts, target -> 전처리
3. max features
4. Sparse matrix
5. train/test split
6. numpy binary file save
"""
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np



# 1. csv file read
spam_data = pd.read_csv('../data/temp_spam_data2.csv', header=None)
spam_data.info() # RangeIndex: 5574 entries, 0 to 5573
spam_data
'''   y변수                      x변수
         0                                                  1
0      ham  Go until jurong point, crazy.. Available only ...
1      ham                      Ok lar... Joking wif u oni...
2     spam  Free entry in 2 a wkly comp to win FA Cup fina...
3      ham  U dun say so early hor... U c already then say...
4      ham  Nah I don't think he goes to usf, he lives aro...
   ...                                                ...
5569  spam  This is the 2nd time we have tried 2 contact u...
5570   ham                Will  b going to esplanade fr home?
5571   ham  Pity, * was in mood for that. So...any other s...
5572   ham  The guy did some bitching but I acted like i'd...
5573   ham                         Rofl. Its true to its name
'''
target = spam_data[0] 
texts = spam_data[1]
target # y변수
texts # x변수
target.shape # (5574,)
texts.shape # (5574,)



# 2. texts, target -> 전처리
# 2-1) target 전처리 -> dummy변수
target = [1 if t == 'spam' else 0 for t in target]
target # 0 ~ 1
# 2-2) texts 전처리
# 별도로해줄필요없음. TfidfVectorizer 함수가해줌.



# 3. max features  :  사용할 x변수의 갯수(열의 차수)
tfidf_fit = TfidfVectorizer().fit(texts) #이거랑 똑같음
vocs = tfidf_fit.vocabulary_
vocs
len(vocs) # 8722
max_features = 4000 
# max_features = 4000 라고하면 8722개 단어중 4000개 단어만 이용 
# sparse matrix = [5574 x 4000]



# 4. Sparse matrix   (max_features =4000 )
sparse_mat = TfidfVectorizer(stop_words='english', 
                              max_features = max_features).fit_transform(texts)
sparse_mat # <5574x4000 sparse matrix of type '<class 'numpy.float64'>'
            # with 39080 stored elements in Compressed Sparse Row format>
print(sparse_mat)
'''
  (0, 3827)     0.22589839945445928
  (0, 1640)     0.18954016110208324
  (0, 919)      0.3415462652078248
  (0, 771)      0.3859368913710106
  (0, 2038)     0.3415462652078248
  :     :
  (5572, 2106)  0.22117089980876273
  (5572, 3849)  0.259646325272169
  (5572, 1516)  0.22117089980876273
  (5573, 3001)  0.7946902152825605
  (5573, 3638)  0.6070152071687805
'''



# 5. scipy -> numpy
sparse_mat_arr = sparse_mat.toarray()
sparse_mat_arr
'''
array([[0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       ...,
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.]])
'''
sparse_mat_arr.shape # (5574, 4000)



# 6. train/test split (70:30)
x_train, x_test, y_train, y_test = train_test_split(sparse_mat_arr, target, 
                                                    test_size=0.3)
x_train.shape # (3901, 4000)
x_test.shape # (1673, 4000)
# 4개를 하나로 묶어주기
spam_data_split = (x_train, x_test, y_train, y_test)



# 7. numpy binary file save

# 7-1) file save 
np.save('../data/spam_data', spam_data_split) # 생략시 : allow_pickle=True 


# 7-2) file load
x_train2, x_test2, y_train2, y_test2 = np.load('../data/spam_data.npy', 
                                               allow_pickle=True)
x_train2.shape # (3901, 4000)
x_test2.shape # (1673, 4000)
