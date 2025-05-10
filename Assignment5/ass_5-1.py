import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import impute
from sklearn import preprocessing
# Given numpy data #
np.random.seed(42)
n = 150
# Synthetic features
age = np.random.normal(40, 10, n)
income = np.random.normal(60000, 15000, n)
purchases = np.random.exponential(300, n)
clicks = np.random.poisson(5, n)
# Inject missing values
income[5] = np.nan
purchases[10] = np.nan
# Inject outliers
income[7] = 300000
purchases[3] = 5000


# Handling missing value #

# # Before impute income and purchases
# print("income\n",income)
# print("purchases\n",purchases)

incomeImputer=impute.SimpleImputer(missing_values=np.nan, strategy='mean')
purchaseImputer=impute.SimpleImputer(missing_values=np.nan, strategy='median')
imputedIncome=(incomeImputer.fit_transform(income.reshape(-1,1))).reshape(150)
imputedPurchases=(purchaseImputer.fit_transform(purchases.reshape(-1,1))).reshape(150)

# # After impute income and purchases
# print(f"income\n{imputedIncome.reshape(150)}")
# print("purchases\n",imputedPurchases.reshape(150))


# Handling outlier #

# # create boxplot with income and purchase
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,16))
# axes[0].boxplot(imputedIncome)
# axes[0].set_title("income boxplot")
# axes[1].boxplot(imputedPurchases)
# axes[1].set_title("purchases boxplot")
# plt.show()

# # Function to replace outlier with closest value
# def replaceOutlier(index, outlier, numpy):
#     min=np.inf
#     replace=0
#     for elm in numpy:
#         if((np.abs(outlier-elm)<min) & (outlier!=elm)):
#             min=np.abs(elm-outlier)
#             replace=elm
#     numpy[index]=replace

# # Detect and replace outlier in income
# Q1=np.percentile(imputedIncome, 25)
# Q3=np.percentile(imputedIncome, 75)
# IQR=Q3-Q1
# outlier=np.where((imputedIncome<Q1-1.5*IQR) | (Q3+1.5*IQR<imputedIncome))[0]
# outlierHandle_income=income.copy()
# print("\noutlier in income")
# for i,inc in enumerate(outlierHandle_income):
#     if(inc<Q1-1.5*IQR):
#         print(f"idx : {i}  val : {inc}")
#         replaceOutlier(i,inc,outlierHandle_income)
#     if(Q3+1.5*IQR<inc):
#         print(f"idx : {i}  val : {inc}")
#         replaceOutlier(i,inc,outlierHandle_income)
# print("\nreplace outlier in income")
# for i in outlier:
#     print(f"idx : {i}  val : {outlierHandle_income[i]}")


# Feture scaling #

# Setting tools for feature scaling
minMaxScaler=preprocessing.MinMaxScaler()
zScoreScaler=preprocessing.StandardScaler()
robustScaler=preprocessing.RobustScaler()
norm=np.sqrt(age**2  + imputedPurchases**2+ clicks**2)

# Feature scaling
minMax_age=(minMaxScaler.fit_transform(age.reshape(-1,1))).reshape(150)
zScore_income=(zScoreScaler.fit_transform(imputedIncome.reshape(-1,1))).reshape(150)
log_purchases=np.log10(imputedPurchases)
robust_income=(robustScaler.fit_transform(imputedIncome.reshape(-1,1))).reshape(150)
vector_features=pd.DataFrame({
    'age' : pd.Series(age/norm),
    'purchases' : pd.Series(imputedPurchases/norm),
    'clicks' : pd.Series(clicks/norm)
    })

# Print result
print("Min-max scaled age\n", minMax_age)
print("Z-scored scaled income\n", zScore_income)
print("Log transformed purchases\n", log_purchases)
print("Robust scaled income\n", robust_income)
print("Vector scaled [age, income, clicks]\n", vector_features)

# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
# axes[0].hist(imputedPurchases, bins=10)
# axes[0].set_title("Histogram of Purchases before scaling")
# axes[1].hist(log_purchases, bins=10)
# axes[1].set_title("Histogram of Purchases after scaling")
# plt.tight_layout()
# plt.show()

fig, axes=plt.subplots(nrows=1, ncols=2, figsize=(8,4))
axes[0].boxplot(imputedIncome, vert=False)
axes[0].set_title("Boxplot of income before scaling")
axes[1].boxplot(robust_income, vert=False)
axes[1].set_title("Boxplot of income after scaling")
plt.tight_layout()
plt.show()