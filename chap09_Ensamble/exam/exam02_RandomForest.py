'''
 문) 당료병(diabetes.csv) 데이터 셋을 이용하여 다음과 같은 단계로 
     RandomForest 모델을 생성하시오.

  <단계1> 데이터셋 로드
  <단계2> x,y 변수 생성 : y변수 : 9번째 칼럼, x변수 : 1 ~ 8번째 칼럼
  <단계3> 500개의 트리를 random으로 생성하여 모델 생성 
  <단계4> 5겹 교차검정/평균 분류정확도 출력
  <단계5> 중요변수 시각화 
'''

from sklearn import model_selection  # cross_validate
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt # 중요변수 시각화 



# 단계1. 테이터셋 로드  
dia = pd.read_csv('C:\\ITWILL\\Work\\4_Python-II\\data/diabetes.csv', header=None) # 제목 없음  
dia.info()  # index 8이 y변수 : 0,1 범주형
print(dia.head()) 

feature_names = list(dia.columns)[:8]
feature_names  # 어차피 숫자라 딱히 필요는 없음


# 단계2. x,y 변수 생성 
y = dia.iloc[:,8]
y
len(y)  # 759
X = dia.iloc[:, :8]
X.shape  # (759, 8)



# 단계3. model 생성
rf = RandomForestClassifier(n_estimators=500)
model = rf.fit(X,y)

pred = model.predict(X)
pred 
pred2 = model.predict_proba(X)
pred2
pred2_dit = pred2.argmax(axis=1)
pred2_dit



# 단계4. 교차검정 model 예측/평가 
cross = model_selection.cross_validate(model, X, y, scoring='accuracy', cv=5)
cross  # 'test_score': array([0.74342105, 0.71710526, 0.73684211, 0.80263158, 0.73509934])
cross_score = cross['test_score']
cross_score.mean()  # 0.7470198675496689



# 단계5. 중요변수 시각화 
# 중요변수
print('중요변수 :', model.feature_importances_)
# 중요변수 : [0.07774522 0.26471768 0.08640061 0.0775544  0.08475314 0.169858 0.12762721 0.11134374]

# 시각화
import matplotlib.pyplot as plt

x_size = X.shape[1]
x_size  # 8

plt.barh(range(x_size), model.feature_importances_)
plt.yticks(range(x_size), feature_names)
plt.xlabel('importance')
plt.show()


































