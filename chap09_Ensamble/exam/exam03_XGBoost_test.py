'''
 문) iris dataset을 이용하여 다음과 같은 단계로 XGBoost model을 생성하시오.
'''

import pandas as pd # file read
from xgboost import XGBClassifier # model 생성 
from xgboost import plot_importance # 중요변수 시각화  
import matplotlib.pyplot as plt # 중요변수 시각화 
from sklearn.model_selection import train_test_split # dataset split
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report # model 평가 
import matplotlib.pyplot as plt


# 단계1 : data set load 
iris = pd.read_csv("C:\\ITWILL\\Work\\4_Python-II\\data/iris.csv")




# 변수명 추출 
cols=list(iris.columns)
col_x=cols[:4] # x변수명 
col_y=cols[-1] # y변수명 

# 단계2 : 훈련/검정 데이터셋 생성
train, test = train_test_split(iris, test_size=0.25, random_state=123)


# 단계3 : model 생성 : train data 이용
xgb = XGBClassifier()
model = xgb.fit(train[col_x], train[col_y])
model
'''
objective='multi:softprob'
max_depth=6,
gamma=0
n_estimators=100
'''


# 단계4 :예측치 생성 : test data 이용  
y_pred = model.predict(test[col_x])
y_pred
y_pred2 = model.predict_proba(test[col_x])
y_pred2
'''
array([[1.21860649e-03, 9.89613950e-01, 9.16750915e-03],   두번째
       [1.19957654e-03, 1.48658839e-03, 9.97313797e-01],   세번째
       [4.42029414e-04, 8.83901550e-04, 9.98674154e-01],   세번째
       .
       .
       .
'''
y_pred3 = y_pred2.argmax(axis=1)
y_pred3  # y_pred 와 같음


# 단계5 : 중요변수 확인 & 시각화 
model.feature_importances_    # array([0.01399811, 0.01935848, 0.41083455, 0.5558089 ], dtype=float32)

fscore = model.get_booster().get_fscore()
fscore  # {'Petal.Length': 96, 'Petal.Width': 66, 'Sepal.Length': 76, 'Sepal.Width': 72}

plot_importance(model, xlabel='F score', ylabel='Features')
plt.show()

feature_im = test[3]


# 단계6 : model 평가 : confusion matrix, accuracy, report
acc = accuracy_score(test[col_y], y_pred)
acc  # 0.9210526315789473

conmat = confusion_matrix(test[col_y], y_pred)
conmat
'''
array([[16,  0,  0],
       [ 0,  8,  0],
       [ 0,  3, 11]], dtype=int64)
'''


report = classification_report(test[col_y], y_pred)
print(report)
'''
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        16
  versicolor       0.73      1.00      0.84         8
   virginica       1.00      0.79      0.88        14

    accuracy                           0.92        38
   macro avg       0.91      0.93      0.91        38
weighted avg       0.94      0.92      0.92        38
'''

#plt.figure(figsize=(5,5))
plt.scatter(test.iloc[:,3], y_pred, s=100, c='r', marker='o')
plt.scatter(test.iloc[:,3], test[col_y], s=50, c='b', marker='o')
plt.legend(loc='best')
plt.show()


