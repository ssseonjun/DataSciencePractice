import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# x=np.arange(1,11)
# y=np.array([20.6, 30.8,55.0,71.4,97.3,131.8,156.3,197.3,238.7,291.7])
# #Pipeline: 처리 단계를 순차적으로 연결해줌
# #poly 단계에서 절편 b0가 만들어지기 때문에
# #linear 단계에서 intercept를 false로 설정
# model=Pipeline([('poly',PolynomialFeatures(degree=2)),
#                 ('linear',LinearRegression(fit_intercept=False))])
# #input x와 output y를 통해 패턴 학습
# model=model.fit(x[:,np.newaxis],y)
# px=np.arange(0,10,0.05)
# #px에 대응되는 결과 예측치를 py에 저장
# py=model.predict(px[:,np.newaxis])
# plt.scatter(x,y)
# plt.plot(px,py,color='r')
# plt.show()

#preprocessing#
#standard normal distribution으로 스케일링_1
from sklearn import preprocessing
score=np.array([20,15,26,32,18,28,35,14,26,22,17], dtype=float)
score_scale=preprocessing.scale(score)
print(score_scale)

#standard normal distribution으로 스케일링_2
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
#fit_transform도 2차원 데이터를 인자로 전달해야 함
#(-1,1)로 reshape -> 열벡터로 전환환
#fit 먼저하고 transform을 여러 번 해야하는 경우 이 방법 사용
score_scaled=scaler.fit_transform(score.reshape(-1,1)).flatten()
print(score_scaled)

#MinMaxScaler: min=0, max=1 로 설정하고 스케일링 함
score=np.array([20,15,26,32,18,28,35,14,26,22,17])
score=score-np.median(score)
minMaxScaler=preprocessing.MinMaxScaler()
score_minmax=minMaxScaler.fit_transform(score[:,np.newaxis]).reshape(-1)
print(score_minmax)

#MaxAbsScaler: max val로 나눠서 (-1,1)의 값으로 스케일링
maxAbsScaler=preprocessing.MaxAbsScaler()
score_maxabs=maxAbsScaler.fit_transform(score[:,np.newaxis]).reshape(-1)
print(score_maxabs)

#Categorical data#
enc=preprocessing.OrdinalEncoder()
x=[['male','from US','uses Safari'],
   ['female','from Europe','uses Firefox'],
      ['female','from Asia','uses chrome']]
#OrdinalEncoding을 기준으로 카테고리 처리를 학습
enc.fit(x)
#카테고리 데이터를 OrdinalEncoding 진행
print(enc.transform(
    [['female','from Europe','uses Firefox'],
      ['male','from Asia','uses chrome']]
))

#OneHotEncoder#
enc=preprocessing.OneHotEncoder()
enc.fit(x)
print(enc.transform([['female','from Europe','uses Firefox'],
      ['male','from Asia','uses chrome']]))