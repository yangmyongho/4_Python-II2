# -*- coding: utf-8 -*-
"""
비교적 단위가 큰 경우 : class
"""
# from module import function
from step01_kNN_data import data_set
import numpy as np

# dataset 생성
know, not_know, cate = data_set()

class kNNClassify : # 클래스는 괄호 없이
    # 생성자,  멤버(메서드, 변수)
    
    def classfy(self, know, not_know, cate, k=3): # 클래스 안의 함수들을 메서드라고 한다.
        # 거리계산식
        diff = know -not_know
        diff
        square_diff = diff ** 2
        square_diff
        sum_square_diff = square_diff.sum(axis = 1) # 행단위 합계
        sum_square_diff 
        distance = np.sqrt(sum_square_diff)
        
        # 단계2 : 오름차순 정렬 -> index
        sortDist = distance.argsort() 
        
        # 단계3 : 최근접 이웃(k=3)
        self.class_result = {} # 빈 set # vote에 이용하기 위해 self의 멤버변수로 만듦.
        
        for i in range(k): # k=3(0~2)
            key = cate[sortDist[i]]
            self.class_result[key] = self.class_result.get(key, 0) + 1
    
    def vote(self):
        vote_re = max(self.class_result)
        print('분류결과 : ', vote_re)
        
# 객체 생성(클래스의 객체 생성 ): 생성자 이용
knn = kNNClassify()
knn.classfy(know, not_know, cate) # class_result 생성된다.
knn.class_result # {'B': 2, 'A': 1} # self의 멤버변수
knn.vote() # 메소드 호출 # 정의에서 인수 없으므로 빈 괄호로 호출하면 된다.

from step01_kNN_data import data_set
import numpy as np

know, not_know, cate = data_set()
know
not_know
cate

class kNNClassify :
    def classify(self, know, not_know, cate, k=3) :
        diff = know-not_know
        square_diff = diff*2
        sum_square_diff = square_diff.sum(axis=1)
        distance = np.sqrt(sum_square_diff)
        
        sortDist = distance.argsort()
        print(sortDist)
        
        self.class_result = {}
        
        for i in range(k) :
            key = cate[sortDist[i]]
            print(key)
            self.class_result[key] = self.class_result.get(key, 0) + 1
            
    def vote(self) :
        vote_re = max(self.class_result)
        print('분류결과 :', vote_re)
            
knn = kNNClassify()            
knn.classify(know, not_know, cate)        
knn.class_result    
knn.vote()
            