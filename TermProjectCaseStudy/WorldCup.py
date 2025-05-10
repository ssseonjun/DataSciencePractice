# step1 Import the Libraries #
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.impute as impute
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
dataset=pd.read_csv(r"C:\Gachon\25_1\DS\Practice_python\TermProjectCaseStudy\WorldCupMatches.csv")

# step2 Dataset #
print(f"head\n{dataset.head()}\n")
print(f"shape\n{dataset.shape}\n")
print(f"index\n{dataset.index}\n")
print(f"columns\n{dataset.columns}\n")

# step3 Delete missing value #

# 아무것도 없는 sample 제거 #
print("Before delete missing values\n")
print(f"# of nullVal\n{dataset.isnull().sum()}\n") #number of null value for each columns
print(f"shape: {dataset.shape}\n")# shape
dataset=dataset.dropna(how='all') #drop sample
print("After delete missing values\n")
# column별 missing value 개수 및 shape #
print(f"# of nullVal\n{dataset.isnull().sum()}\n") #find missing value for each columns, Attendance 2개
print(f"shape: {dataset.shape}\n") #shape


# mean값 계산하기기#
print("Calculate mean value")
mean=np.mean(dataset['Attendance'])
print(f"mean of 'Attendance': {mean}\n")

# mean으로 missing value 채우기 #
print("Handling missing value\n")
df1=dataset.copy()
index=df1['Attendance'].isna().to_numpy() # Attendance가 null인 sample의 인덱스 /  엑셀기준 825, 843번째 data
print(f"Before impute\n{df1.loc[index,'Attendance']}\n")
df1['Attendance']=df1['Attendance'].replace(np.nan,mean)
print(f"After impute\n{df1.loc[index,'Attendance']}\n")

# simple imputer로 missing value 채우기 #
print("Handling missing value with simple imputer\n")
df2=dataset.copy()
index=df2['Attendance'].isna().to_numpy() # Attendance가 null인 sample의 인덱스 /  엑셀기준 825, 843번째 data
print(f"before impute\n{df2.loc[index,'Attendance']}\n")
# simple impute #
simpleImputer=impute.SimpleImputer(missing_values=np.nan,strategy='mean')
simpleImputer.fit(df2[['Attendance']])
imputed_Attendance=simpleImputer.transform(df2[['Attendance']])
# mean값 입력 #
df2['Attendance']=imputed_Attendance.reshape(-1)
print(f"After impute\n{df2.loc[index,'Attendance']}\n")