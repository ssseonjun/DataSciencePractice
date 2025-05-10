import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import linear_model
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
import matplotlib.pyplot as plt

# Part 1 #
cancer=load_breast_cancer()
print(dir(cancer))
data=cancer.data
features=cancer.feature_names
target=cancer.target

print(f"Number of samples\n{len(data)}\n")
print(f"Feature names\n{features}\n")
print(f"Target class distributionn\n{cancer.target}\n[0, 1] : {cancer.target_names}\n")

minMaxResult=preprocessing.MinMaxScaler()
dataResult=minMaxResult.fit_transform(data)
print(f"Min-Max scale all features\n{dataResult}\n")


# Part 2 #

# # Chi-squared test
# df=pd.DataFrame(dataResult, columns=features)
# modelChi=SelectKBest(score_func=chi2, k=len(features))
# modelChi.fit(df,target)
# result=pd.Series(modelChi.scores_,index=features)
# sort_result=result.sort_values(ascending=True)
# five_result=sort_result.tail()
# plt.figure(figsize=(8,6))
# plt.barh(sort_result.index,sort_result, color='lemonchiffon')
# plt.barh(five_result.index,five_result, color='goldenrod')
# plt.tight_layout()
# plt.show()

# Lasso Regression
modelLasso=linear_model.Lasso(alpha=0.01)
modelLasso.fit(dataResult,target)
coef=pd.Series(modelLasso.coef_,index=features)
sort_coef=coef.sort_values(ascending=True)
five_coef=sort_coef[sort_coef!=0]
print(f"Result of lasso regression\n{sort_coef}\n")
print(f"Selected features\n{five_coef}\n")

# # Tree-based Model
# x=pd.DataFrame(dataResult,columns=features)
# y=target

# modelTree=ExtraTreesClassifier(n_estimators=100,random_state=42)
# modelTree.fit(x,y)

# result=pd.Series(modelTree.feature_importances_,index=x.columns)
# result_sorted=result.sort_values(ascending=True)
# result_five=result_sorted.tail()
# print(result_five)
# plt.figure(figsize=(8,6))
# plt.barh(result_sorted.index,result_sorted,color='mistyrose')
# plt.barh(result_five.index,result_five,color='salmon')
# plt.title("Feature importance sorted by tree")
# plt.xlabel("Importance Score")
# plt.tight_layout()
# plt.show()

