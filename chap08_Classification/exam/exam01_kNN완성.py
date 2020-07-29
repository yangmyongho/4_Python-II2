'''
문) 다음과 같은 3개의 범주를 갖는 6개의 데이터 셋을 대상으로 kNN 알고리즘을 적용하여 
      특정 품목을 분류하시오.
   (단계1 : 함수 구현  -> 단계2 : 클래스 구현)  
      
    <조건1> 데이터 셋  
    -------------------------
     품목     단맛 아삭거림 분류범주
    -------------------------
    grape   8   5     과일
    fish    2   3     단백질 
    carrot  7   10    채소
    orange  7   3     과일 
    celery  3   8     채소
    cheese  1   1     단백질 
    ------------------------
    
   <조건2> 분류 대상과 k값은 키보드 입력  
   
  <<출력 예시 1>> k=3인 경우
  -----------------------------------
    단맛 입력(1~10) : 8
    아삭거림 입력(1~10) : 4
  k값 입력(1 or 3) : 3
  -----------------------------------
  calssCount: {'과일': 2, '단백질': 1}
   분류결과: 과일
  -----------------------------------
  
  <<출력 예시 2>> k=1인 경우
  -----------------------------------
   단맛 입력(1~10) : 2
   아삭거림 입력(1~10) :3
  k값 입력(1 or 3) : 1
  -----------------------------------
  calssCount: {'단백질': 1}
   분류결과 : 단백질
  -----------------------------------
'''
import numpy as np

grape = [8, 5]
fish = [2, 3]
carrot = [7, 10]
orange = [7, 3]
celery = [3, 8]
cheese = [1, 1]
class_category = ['과일', '단백질', '채소', '과일', '채소', '단백질']

# data 생성 함수 정의
def data_set():
    # 선형대수 연산 : numpy형 변환 
    know_group = np.array([grape, fish, carrot, orange,celery,cheese]) # 알려진 집단 - 2차원 
    class_cate = np.array(class_category) # 정답(분류범주)
    return know_group, class_cate 

know, cate = data_set()
know
cate

# not_know, k -> 키보드 입력 
def classify(know, not_know, cate, k) :
    # 단계1 : 거리계산식 
    diff =  know - not_know  
    square_diff = diff ** 2    
    sum_square_diff = square_diff.sum(axis = 1) # 행 단위 합계    
    distance = np.sqrt(sum_square_diff)
    
    # 단계2 : 오름차순 정렬 -> index 
    sortDist = distance.argsort()
    
    # 단계3 : 최근접 이웃(k=3)
    class_result = {} # 빈 set 
    
    for i in range(k) : # k = 3(0~2)
        key = cate[sortDist[i]]
        class_result[key] = class_result.get(key, 0) + 1
        
    return class_result # {'B': 2, 'A': 1}

# 키보드 입력 
x = int(input('단맛 입력(1~10) :'))
y = int(input('아삭거림 입력(1~10) :'))
k = int(input('k값 입력(1 or 3) :'))

# 알려지지 않은 집단 
not_know = np.array([x, y])

# 함수 호출 
class_result = classify(know, not_know, cate, k)
print('-'*40)
print('calssCount:', class_result)
print('분류결과 :',max(class_result, key=class_result.get)) # value 기준 
print('-'*40) 



