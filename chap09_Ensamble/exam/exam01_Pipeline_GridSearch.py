'''
 문) digits 데이터 셋을 이용하여 다음과 단계로 Pipeline 모델을 생성하시오.
  <단계1> dataset load
  <단계2> Pipeline model 생성
          - scaling : StndardScaler 클래스, modeing : SVC 클래스    
  <단계3> GridSearch model 생성
          - params : 10e-2 ~ 10e+2, 평가방법 : accuracy, 교차검정 : 5겹
          - CPU 코어 수 : 2개 
  <단계4> best score, best params 출력 
'''

from sklearn.datasets import load_digits # dataset 
from sklearn.svm import SVC # model
from sklearn.model_selection import GridSearchCV # gride search model
from sklearn.pipeline import Pipeline # pipeline
from sklearn.preprocessing import StandardScaler # dataset scaling
from sklearn.model_selection import train_test_split


# 1. dataset load
digits = load_digits()
print(digits.DESCR)
X = digits.data
y = digits.target
X.shape  #  (1797, 64)
X.min()  # 0.0
X.max()  #  16.0  >> put into scaler

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)


# 2. pipeline model : step01(data 표준화) -> stpe02(model)
pipe_svm = Pipeline([ ('scaler', StandardScaler()), ('svc', SVC(gamma='auto')) ])
pmodel = pipe_svm.fit(x_train, y_train)

score = pmodel.score(x_test, y_test)
score  # 0.9814814814814815



# 3. grid search model 
params = [0.01, 0.1, 1.0, 10.0, 100.0]
params_grid = [ {'svc__C':params, 'svc__kernel':['linear']},
               {'svc__C':params, 'svc__gamma':params, 'svc__kernel':['rbf']} ]

gs = GridSearchCV(pipe_svm, params_grid, scoring='accuracy', cv=5, n_jobs=2)
gsmodel = gs.fit(x_train, y_train)

score = gsmodel.score(x_test, y_test)
score  # 0.975925925925926  : 점수가 다르다. 왜????????????????????????????????????????

# 교차검정 결과
gsmodel.cv_results_["mean_test_score"]
'''
array([0.97533359, 0.98488902, 0.98488585, 0.98488585, 0.98488585,
       0.10898628, 0.10898628, 0.10898628, 0.10898628, 0.10898628,
       0.92839752, 0.21797888, 0.10898628, 0.10898628, 0.10898628,
       0.97692405, 0.94191488, 0.11455448, 0.10898628, 0.10898628,
       0.98249225, 0.94509581, 0.1185259 , 0.10898628, 0.10898628,
       0.98249225, 0.94509581, 0.1185259 , 0.10898628, 0.10898628])
'''



# 4. best score, best params
gsmodel.best_params_  #  {'svc__C': 0.1, 'svc__kernel': 'linear'}
gsmodel.best_score_  # 0.984889015367103

































