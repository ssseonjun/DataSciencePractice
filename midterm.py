# import numpy as np
# import matplotlib.pyplot as plt

# np.random.seed(42)
# data=np.concatenate([
#     np.random.normal(loc=50,scale=10,size=100),
#     np.array([10,110])
# ])

# Q1=np.percentile(data,25)
# Q3=np.percentile(data,75)
# IQR=Q3-Q1

# lower_bound=Q1-1.5*IQR
# upper_bound=Q3+1.5*IQR

# outliers=data[(data<lower_bound)|(upper_bound<data)]

# fig=plt.figure(figsize=(8,4))
# plt.boxplot(data,vert=False,patch_artist=True,boxprops=dict(facecolor='lightblue'))
# plt.title("Box Plot with Outliers")
# plt.xlabel("Value")

# for outlier in outliers:
#     plt.plot(outlier,1,'ro')
#     #ha: horizental alignment, 수평위치 조절
#     plt.text(outlier,1.05,f"{outlier:.1f}",color='red',ha='center', fontsize=9)

# #x축에 맞게 수직격자 생성, alpha : 격자의 투명도
# plt.grid(True,axis='x',linestyle='--',alpha=0.6)
# #figsize를 유지하면서 여백을 조절,
# #그래프와 요소들의 위치를 최적화시켜줌
# plt.tight_layout()
# plt.show()

import numpy as np
import random as random
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import impute

import pandas as pd

A=np.ones((3,4,5,6))
A=A[...,np.newaxis,]
print(A.shape)