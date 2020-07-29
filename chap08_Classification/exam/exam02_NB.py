'''
문) weatherAUS.csv 파일을 시용하여 NB 모델을 생성하시오
  조건1> NaN 값을 가진 모든 row 삭제 
  조건2> 1,2,8,10,11,22,23 칼럼 제외 
  조건3> 7:3 비율 train/test 데이터셋 구성 
  조건4> formula 구성  = RainTomorrow ~ 나머지 변수(16개)
  조건5> GaussianNB 모델 생성 
  조건6> model 평가 : accuracy, confusion matrix, f1 score
'''
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


data = pd.read_csv('C:\\ITWILL\\Work\\4_Python-II\\data/weatherAUS.csv')
print(data.head())
print(data.info())



# 조건2> 1,2,8,10,11,22,23 칼럼 제외 
col = list(data.columns)
col = list(data.columns)
col
for i in [1,2,8,10,11,22,23] :
    col.remove(list(data.columns)[i-1])
print(col)

# dataset 생성 
new_data = data[col]
new_data = data[col]
print(new_data.head())
print(new_data.head())
new_data.info()
new_data['RainTomorrow']




# 조건1> NaN 값을 가진 모든 row 삭제
data=data.dropna()
data = data.dropna()

print(data.head())





# 조건3> 7:3 비율 train/test 데이터셋 구성
train, test = train_test_split(
new_data, test_size=0.3, random_state=0) # seed값 



# 조건4> formula 구성  = RainTomorrow ~ 나머지 변수(16개)
len(col)
col.index
x_cols = col[:16]
y_col = col[-1]

type(y_col)  # str


# formula = 'RainTomorrow ~ MinTemp + MaxTemp + Rainfall + Evaporation + Sunshine + WindGustSpeed+ WindSpeed9am + WindSpeed3pm + Humidity9am + Humidity3pm + Pressure9am + Pressure3pm + Cloud9am +Cloud3pm +Temp9am+Temp3pm'
# # 어따가 쓰라는건지.....****************************************



# 조건5> GaussianNB 모델 생성 
model = GaussianNB().fit(X = train[x_cols], y = train[y_col])



# 조건6> model 평가 : accuracy, confusion matrix, f1 score
y_pred = model.predict(test[x_cols])
y_true = test[y_col]

acc = accuracy_score(y_true, y_pred)
acc  #  0.8051400076716533

conmat = confusion_matrix(y_true, y_pred)
conmat
'''array([[3391,  676],
       [ 340,  807]], dtype=int64)'''
acc2 = (conmat[0,0] + conmat[1,1]) / conmat.sum()
acc2  #  0.8051400076716533

acc3 = f1_score(y_true, y_pred, average='micro')
acc3
  # 0.8051400076716534  매우 적은 차이가 남.

acc4 = f1_score(y_true, y_pred, average='macro')
acc4  # 0.7416991975128653
