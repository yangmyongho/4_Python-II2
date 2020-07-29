# -*- coding: utf-8 -*-
"""
step02_TfidfSparseMatrix

1. csv file read : temp_spam_data.csv
2. texts, target -> 전처리
3. max features
4. Sparse matrix
"""
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import string # text 전처리



# 1. csv file read
spam_data = pd.read_csv('../data/temp_spam_data.csv', encoding='utf-8',
                        header=None)
spam_data
'''  y변수                    x변수
      0                        1
0   ham    우리나라    대한민국, 우리나라 만세
1  spam      비아그라 500GRAM 정력 최고!
2   ham               나는 대한민국 사람
3  spam  보험료 15000원에 평생 보장 마감 임박
4   ham                   나는 홍길동
'''
target = spam_data[0] 
texts = spam_data[1]
target # y변수
texts # x변수
target.shape # (5,)
texts.shape # (5,)



# 2. texts, target -> 전처리

# 2-1) target 전처리 -> dummy변수
target = [1 if t == 'spam' else 0 for t in target]
target # [0, 1, 0, 1, 0]


# 2-2) texts 전처리
def text_prepro(texts): # 문단(input) -> 문장(string) -> 음절 -> 문장
    # Lower case(소문자변경)
    texts = [x.lower() for x in texts]
    # Remove punctuation(특수문자,문장부호제거)
    texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]
    # Remove numbers(숫자제거)
    texts = [''.join(c for c in x if c not in string.digits) for x in texts]
    # Trim extra whitespace(띄어쓰기생성)
    texts = [' '.join(x.split()) for x in texts]
    return texts
texts_re = text_prepro(texts)
texts_re
'''
['우리나라 대한민국 우리나라 만세',
 '비아그라 gram 정력 최고',
 '나는 대한민국 사람',
 '보험료 원에 평생 보장 마감 임박',
 '나는 홍길동']
'''



# 3. max features  :  사용할 x변수의 갯수(열의 차수)
tfidf = TfidfVectorizer()
tfidf_fit = tfidf.fit(texts_re) # 문장 -> 단어 생성
# = tfidf_fit = TfidfVectorizer().fit(texts_re) 이거랑 똑같음
vocs = tfidf_fit.vocabulary_
vocs
# max_features = len(vocs) # 16  -> 생략했을경우
max_features = 10
''' 만약 max_features = 10 라고하면 16개 단어중 10개 단어만 이용
sparse matrix = [5 x 10] '''



# 4. Sparse matrix 

# 4-1) max_features 생략했을때
sparse_mat = TfidfVectorizer().fit_transform(texts_re)
sparse_mat # <5x16 sparse matrix of type '<class 'numpy.float64'>'
print(sparse_mat)
'''
  (0, 4)        0.4206690600631704
  (0, 2)        0.3393931489111758
  (0, 9)        0.8413381201263408
  (1, 13)       0.5
  (1, 12)       0.5
  (1, 0)        0.5
  (1, 7)        0.5
  (2, 8)        0.6591180018251055
  (2, 1)        0.5317722537280788
  (2, 2)        0.5317722537280788
  (3, 11)       0.40824829046386296
  (3, 3)        0.40824829046386296
  (3, 5)        0.40824829046386296
  (3, 14)       0.40824829046386296
  (3, 10)       0.40824829046386296
  (3, 6)        0.40824829046386296
  (4, 15)       0.7782829228046183
  (4, 1)        0.6279137616509933
'''


#  4-2) max_features =10 적용했을때
sparse_mat2 = TfidfVectorizer(max_features = max_features).fit_transform(texts_re)
sparse_mat2 # <5x10 sparse matrix of type '<class 'numpy.float64'>'
print(sparse_mat2)
'''
  (0, 4)        0.4206690600631704
  (0, 2)        0.3393931489111758
  (0, 9)        0.8413381201263408
  (1, 0)        0.7071067811865475
  (1, 7)        0.7071067811865475
  (2, 8)        0.6591180018251055
  (2, 1)        0.5317722537280788
  (2, 2)        0.5317722537280788
  (3, 3)        0.5773502691896258
  (3, 5)        0.5773502691896258
  (3, 6)        0.5773502691896258
  (4, 1)        1.0
'''



# 5. scipy -> numpy
sparse_mat_arr = sparse_mat.toarray()
sparse_mat_arr
'''
array([[0.        , 0.        , 0.33939315, 0.        , 0.42066906,
        0.        , 0.        , 0.        , 0.        , 0.84133812,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.5       , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.5       , 0.        , 0.        ,
        0.        , 0.        , 0.5       , 0.5       , 0.        ,
        0.        ],
       [0.        , 0.53177225, 0.53177225, 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.659118  , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        ],
       [0.        , 0.        , 0.        , 0.40824829, 0.        ,
        0.40824829, 0.40824829, 0.        , 0.        , 0.        ,
        0.40824829, 0.40824829, 0.        , 0.        , 0.40824829,
        0.        ],
       [0.        , 0.62791376, 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.77828292]])
'''





































