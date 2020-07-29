# -*- coding: utf-8 -*-
"""
XGBoost : 회귀트리(XGBRegressor)
"""


# import test
from xgboost import XGBClassifier, XGBRegressor  # model
from xgboost import plot_importance  # x변수에 대한 중요변수 시각화
from sklearn.datasets import load_boston  # 주택가격 데이터셋 생성
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score  # 각각 y변수가 정규화됐을 때, 되지 않았을 때


# 1. dataset load
boston = load_boston()
x_names = boston.feature_names  # x 변수명
# array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'], dtype='<U7')


X, y = load_boston(return_X_y=True)
X.shape  # (506, 13)

y  # 비율척도, 비정규화


# 2. train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



# 3. create model
xgb = XGBRegressor()
model =xgb.fit(x_train, y_train)
model  # objective='reg:squarederror'



# 4. 중요변수 시각화
fscore = model.get_booster().get_fscore()
fscore
'''
{'f5': 378,
 'f12': 254,
 'f0': 642,
 'f4': 135,
 'f7': 238,
 'f11': 289,
 'f8': 27,
 'f1': 60,
 'f3': 15,
 'f10': 64,
 'f6': 318,
 'f9': 72,
 'f2': 122}
'''
x_names[0]  # 'CRIM'  범죄율
x_names[5]  # 'RM'  방의 개수

plot_importance(model)
import matplotlib.pyplot as plt
plt.show()



# 5. model 평가
y_pred = model.predict(x_test)
y_pred

mse = mean_squared_error(y_test, y_pred)
mse  # 19.36792601139511  정규화가 되지 않았으므로 수가 이상하다

r2 = r2_score(y_test, y_pred)
r2  # 0.786859481962882



























