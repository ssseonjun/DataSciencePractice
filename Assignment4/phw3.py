# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn import preprocessing
# from sklearn import linear_model

# df=pd.read_excel(r'C:\Gachon\25_1\DS\Practice_python\bmi_data_phw3.xlsx')
# print(df)

# # sort dataFrame by BMI level #
# bmi_values=sorted((df.sort_values(by='BMI'))['BMI'].unique())

# # Make (height, BMI) and (weight, BMI) histogram #
# fig,axes=plt.subplots(nrows=2,ncols=len(bmi_values),figsize=(10*len(bmi_values),10))
# plt.suptitle('height & weight for each BMI')
# plt.subplots_adjust(hspace=0.4)
# plt.subplots_adjust(wspace=0.7)

# for i,bmi in enumerate(bmi_values):
#     data=df[df['BMI']==bmi]

#     axes[0][i].hist(data['Height (Inches)'],bins=10)
#     axes[0][i].set_title(f'height-BMI={bmi}')
#     axes[0][i].set_xlabel('height')
#     axes[0][i].set_ylabel('count')
    
#     axes[1][i].hist(data['Weight (Pounds)'],bins=10)
#     axes[1][i].set_title(f'weight-BMI={bmi}')
#     axes[1][i].set_xlabel('weight')
#     axes[1][i].set_ylabel('count')

# plt.show()

# # MinMaxScaled #
# df2=df.copy()
# height=df2['Height (Inches)'].to_numpy()
# weight=df2['Weight (Pounds)'].to_numpy()
# model_minMax=preprocessing.MinMaxScaler()
# df2['Height (Inches)']=model_minMax.fit_transform(height.reshape(-1,1))
# df2['Weight (Pounds)']=model_minMax.fit_transform(weight.reshape(-1,1))

# # StandardScaled #
# df3=df.copy()
# height=df3['Height (Inches)'].to_numpy()
# weight=df3['Weight (Pounds)'].to_numpy()
# model_stand=preprocessing.StandardScaler()
# df3['Height (Inches)']=model_stand.fit_transform(height.reshape(-1,1))
# df3['Weight (Pounds)']=model_stand.fit_transform(weight[:,np.newaxis])

# # Robust Scaled #
# df4=df.copy()
# height=df4['Height (Inches)'].to_numpy()
# weight=df4['Weight (Pounds)'].to_numpy()
# model_robust=preprocessing.RobustScaler()
# df4['Height (Inches)']=model_robust.fit_transform(height[:,np.newaxis])
# df4['Weight (Pounds)']=model_robust.fit_transform(weight.reshape(-1,1))

# # Make (height, BMI) and (weight, BMI) histogram using min_max scaled data# 
# fig,axes=plt.subplots(nrows=2,ncols=len(bmi_values),figsize=(10*len(bmi_values),6))
# fig.suptitle('MinMax scaled result for weight & height',fontsize=10)
# fig.subplots_adjust(hspace=0.4)
# fig.subplots_adjust(wspace=0.7)

# for i,bmi in enumerate(bmi_values):
#     data=df2[df2['BMI']==bmi]

#     axes[0][i].hist(data['Height (Inches)'],bins=10)
#     axes[0][i].set_title(f'height-BMI({bmi})')
#     axes[0][i].set_xlabel('height')
#     axes[0][i].set_ylabel('freq')

#     axes[1][i].hist(data['Weight (Pounds)'],bins=10)
#     axes[1][i].set_title(f'weight-BMI({bmi})')
#     axes[1][i].set_xlabel('weight')
#     axes[1][i].set_ylabel('freq')
# plt.show()

# # Make (height, BMI) and (weight, BMI) histogram using standard scaled data# 
# fig,axes=plt.subplots(nrows=2,ncols=len(bmi_values),figsize=(10*len(bmi_values),6))
# fig.suptitle('Standard scaled result for weight & height',fontsize=10)
# fig.subplots_adjust(hspace=0.4)
# fig.subplots_adjust(wspace=0.7)

# for i,bmi in enumerate(bmi_values):
#     data=df3[df3['BMI']==bmi]

#     axes[0][i].hist(data['Height (Inches)'],bins=10)
#     axes[0][i].set_title(f'height-BMI({bmi})')
#     axes[0][i].set_xlabel('height')
#     axes[0][i].set_ylabel('freq')

#     axes[1][i].hist(data['Weight (Pounds)'],bins=10)
#     axes[1][i].set_title(f'weight-BMI({bmi})')
#     axes[1][i].set_xlabel('weight')
#     axes[1][i].set_ylabel('freq')
# plt.show()

# # Make (height, BMI) and (weight, BMI) histogram using robust scaled data# 
# fig,axes=plt.subplots(nrows=2,ncols=len(bmi_values),figsize=(10*len(bmi_values),6))
# fig.suptitle('Robust scaled result for weight & height',fontsize=10)
# fig.subplots_adjust(hspace=0.4)
# fig.subplots_adjust(wspace=0.7)

# for i,bmi in enumerate(bmi_values):
#     data=df4[df4['BMI']==bmi]

#     axes[0][i].hist(data['Height (Inches)'],bins=10)
#     axes[0][i].set_title(f'height-BMI({bmi})')
#     axes[0][i].set_xlabel('height')
#     axes[0][i].set_ylabel('freq')

#     axes[1][i].hist(data['Weight (Pounds)'],bins=10)
#     axes[1][i].set_title(f'weight-BMI({bmi})')
#     axes[1][i].set_xlabel('weight')
#     axes[1][i].set_ylabel('freq')
# plt.show()

# # compute linear regression using height and weight
# # and normalize e value
# df5=df.copy()
# learn=df5[(df5['Height (Inches)'].notna()) & (df5['Weight (Pounds)'].notna())]
# height=learn[['Height (Inches)']]
# weight=learn['Weight (Pounds)']
# test=df5[df5['Height (Inches)'].notna()]
# indices=df5.index[df5['Height (Inches)'].notna()].tolist()

# reg=linear_model.LinearRegression()
# reg.fit(height,weight)
# weightExpected=reg.predict(test[['Height (Inches)']])
# weight=test['Weight (Pounds)']
# e=weight-weightExpected
# Z_scored=(e-np.mean(e))/np.std(e)

# plt.hist(Z_scored,bins=10)
# plt.xlabel('Ze')
# plt.ylabel('frequency')
# plt.show()

# # decide alpha
# alpha=1.3
# print('\ndecide alpha=1.3')
# for idx in indices:
#     if(Z_scored.loc[idx]>alpha):
#         df5.loc[idx,'BMI']=4
#         print(f'{idx}-th BMI level changed to 4')
#     if(Z_scored.loc[idx]<-alpha):
#         df5.loc[idx,'BMI']=0
#         print(f'{idx}-th BMI level changed to 0')

# # compute linear regression using height and weight
# # and normalize e value for male group
# df6=df.copy()
# learn=df6[(df6['Height (Inches)'].notna())&(df6['Weight (Pounds)'].notna())]
# male=learn[learn['Sex']=='Male']
# maleHeightLearn=male[['Height (Inches)']]
# maleWeightLearn=male['Weight (Pounds)']
# test=df6[(df6['Sex']=='Male') & (df6['Height (Inches)'].notna())]
# maleIndices=df6.index[(df6['Sex']=='Male')&(df6['Height (Inches)'].notna())].tolist()

# reg.fit(maleHeightLearn,maleWeightLearn)
# maleWeightExpected=reg.predict(test[['Height (Inches)']])
# maleWeight=test['Weight (Pounds)']
# e=maleWeight-maleWeightExpected
# Z_scored=(e-np.mean(e))/np.std(e)

# plt.hist(Z_scored,bins=10)
# plt.xlabel('Ze')
# plt.ylabel('frequency')
# plt.show()

# # decide alpha
# alpha=1.4
# print('\ndecide alpha=1.4 for male')
# for idx in maleIndices:
#     if(Z_scored.loc[idx]>alpha):
#         df6.loc[idx,'BMI']=4
#         print(f'{idx}-th male BMI level changed to 4')
#     if(Z_scored.loc[idx]<-alpha):
#         df6.loc[idx,'BMI']=0
#         print(f'{idx}-th male BMI level changed to 0')

# # compute linear regression using height and weight
# # and normalize e value for female group
# df6=df.copy()
# female=learn[learn['Sex']=='Female']
# femaleHeightLearn=female[['Height (Inches)']]
# femaleWeightLearn=female['Weight (Pounds)']
# test=df6[(df6['Sex']=='Female') & (df6['Height (Inches)'].notna())]
# femaleIndices=df6.index[(df6['Sex']=='Female')&(df6['Height (Inches)'].notna())].tolist()

# reg.fit(femaleHeightLearn,femaleWeightLearn)
# femaleWeightExpected=reg.predict(test[['Height (Inches)']])
# femaleWeight=test['Weight (Pounds)']
# e=femaleWeight-femaleWeightExpected
# Z_scored=(e-np.mean(e))/np.std(e)

# plt.hist(Z_scored,bins=10)
# plt.xlabel('Ze')
# plt.ylabel('frequency')
# plt.show()

# # decide alpha
# alpha=1.5
# print('\ndecide alpha=1.5 for female')
# for idx in femaleIndices:
#     if(Z_scored.loc[idx]>alpha):
#         df6.loc[idx,'BMI']=4
#         print(f'{idx}-th female BMI level changed to 4')
#     if(Z_scored.loc[idx]<-alpha):
#         df6.loc[idx,'BMI']=0
#         print(f'{idx}-th female BMI level changed to 0')