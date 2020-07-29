'''
외식업종 관련 data set 분석

문) food_df를 대상으로 다음과 같이 xgboost 모델을 생성하시오.
   <조건1> 6:4 비율 train/test set 생성 
   <조건2> y변수 ; 폐업_2년, x변수 ; 나머지 20개 
   <조건3> 중요변수에 대한  f1 score 출력
   <조건4> 중요변수 시각화  
   <조건5> accuracy와 model report 출력 
'''

import pandas as pd
from sklearn import model_selection, metrics
from sklearn.preprocessing import minmax_scale # 정규화 함수 
from xgboost import XGBClassifier # xgboost 모델 생성 
from xgboost import plot_importance # 중요변수 시각화  

# 중요변수 시각화 
from matplotlib import pyplot
from matplotlib import font_manager, rc # 한글 지원
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

# 외식업종 관련 data set
food = pd.read_csv("C:\\ITWILL\Work\\4_Python-II\\data/food_dataset.csv",encoding="utf-8",thousands=',')  # 천 단위 숫자를 1,000 이렇게 숫자로 인식

# 결측치 제거
food=food.dropna()  
print(food.info())  # 이 업종이 2년안에 폐업할 것인지를 20개의 x 변수를 이용해 예측

# minmax scale
food_nor = minmax_scale(food)
food_nor.mean()
food_df = pd.DataFrame(food_nor, columns=food.columns)
food_df.info()



 
#   <조건1> 6:4 비율 train/test set 생성 
train, test = model_selection.train_test_split(food_df, test_size=0.4)

train.shape
train.info()


#   <조건2> y변수 ; 폐업_2년, x변수 ; 나머지 20개 

cols = list(food.columns)

y_col = cols[-1]
food[y_col]  # 0 or 1
x_col = cols[:20]



xgb = XGBClassifier()
model = xgb.fit(train[x_col], train[y_col])
model  # objective='binary:logistic'



#   <조건3> 중요변수에 대한  f1 score 출력
fscore = model.get_booster().get_fscore()
fscore
'''
{'소재지면적': 566!!!!!, '주변': 341, 'bank': 266, 'X3km_대학교갯수': 129, '기간평균': 553, 'tax_sum': 235,
 'pop': 244, '유동인구_주중_오전': 339, 'X1km_초등학교갯수': 115, '유동인구_주말_오전': 288,
 'X1km_지하철역갯수': 77, 'X1km_병원갯수': 145, 'X1km_고등학교갯수': 107, '유동인구_주중_오후': 311,
 '유동인구_주말_오후': 266, 'nonbank': 267, 'X1km_영화관갯수': 101, '주변동종': 223, '위생업태명': 132}
'''



#   <조건4> 중요변수 시각화 
plot_importance(model)
import matplotlib.pyplot as plt
plt.show()




#   <조건5> accuracy와 model report 출력 
y_pred = model.predict(test[x_col])
y_pred

y_pred2 = model.predict_proba(test[x_col])
y_pred2

y_pred3 = y_pred2.argmax(axis=1)
y_pred3


acc = metrics.accuracy_score(test[y_col], y_pred)
acc  # 0.7897452669065009

conmat = metrics.confusion_matrix(test[y_col], y_pred)
conmat
'''
array([[21222,   591],
       [ 5195,   511]], dtype=int64)
'''  # 안망할애들은 예측이 많이 빗나감

f1score = metrics.f1_score(test[y_col], y_pred)
f1score  # 0.150117508813161  :  정확률과 재현률의 조화평균 -> 


score = model.score(test[x_col], test[y_col])
score  # 0.7897452669065009

report = metrics.classification_report(test[y_col], y_pred)
print(report)

'''
              precision    recall  f1-score   support

         0.0       0.80      0.97      0.88     21813
         1.0       0.46      0.09      0.15      5706

    accuracy                           0.79     27519
   macro avg       0.63      0.53      0.52     27519
weighted avg       0.73      0.79      0.73     27519
'''











