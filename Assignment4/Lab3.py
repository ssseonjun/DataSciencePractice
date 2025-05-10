# import numpy as np
# import pandas as  pd
# import matplotlib.pyplot as plt
# from sklearn import preprocessing
# from sklearn import impute

# # Read csv file and print dataFrame #
# df=pd.read_csv(r'C:\Gachon\25_1\DS\Practice_python\bmi_data_lab3.csv')

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
# fig.subplots_adjust(hspace=0.7)
# fig.subplots_adjust(wspace=0.4)

# for i,bmi in enumerate(bmi_values):
#     data=df2[df2['BMI']==bmi]

#     axes[0][i].hist(data['Height (Inches)'],bins=10)
#     axes[0][i].set_title(f'height-BMI={bmi}')
#     axes[0][i].set_xlabel('height')
#     axes[0][i].set_ylabel('freq')

#     axes[1][i].hist(data['Weight (Pounds)'],bins=10)
#     axes[1][i].set_title(f'weight-BMI={bmi}')
#     axes[1][i].set_xlabel('weight')
#     axes[1][i].set_ylabel('freq')
# plt.show()

# # Make (height, BMI) and (weight, BMI) histogram using standard scaled data# 
# fig,axes=plt.subplots(nrows=2,ncols=len(bmi_values),figsize=(10*len(bmi_values),6))
# fig.suptitle('Standard scaled result for weight & height',fontsize=10)
# fig.subplots_adjust(hspace=0.7)
# fig.subplots_adjust(wspace=0.4)

# for i,bmi in enumerate(bmi_values):
#     data=df3[df3['BMI']==bmi]

#     axes[0][i].hist(data['Height (Inches)'],bins=10)
#     axes[0][i].set_title(f'height-BMI={bmi}')
#     axes[0][i].set_xlabel('height')
#     axes[0][i].set_ylabel('freq')

#     axes[1][i].hist(data['Weight (Pounds)'],bins=10)
#     axes[1][i].set_title(f'weight-BMI={bmi}')
#     axes[1][i].set_xlabel('weight')
#     axes[1][i].set_ylabel('freq')
# plt.show()

# # Make (height, BMI) and (weight, BMI) histogram using robust scaled data# 
# fig,axes=plt.subplots(nrows=2,ncols=len(bmi_values),figsize=(10*len(bmi_values),6))
# fig.suptitle('Robust scaled result for weight & height',fontsize=10)
# fig.subplots_adjust(hspace=0.7)
# fig.subplots_adjust(wspace=0.4)

# for i,bmi in enumerate(bmi_values):
#     data=df4[df4['BMI']==bmi]

#     axes[0][i].hist(data['Height (Inches)'],bins=10)
#     axes[0][i].set_title(f'height-BMI={bmi}')
#     axes[0][i].set_xlabel('height')
#     axes[0][i].set_ylabel('freq')

#     axes[1][i].hist(data['Weight (Pounds)'],bins=10)
#     axes[1][i].set_title(f'weight-BMI={bmi}')
#     axes[1][i].set_xlabel('weight')
#     axes[1][i].set_ylabel('freq')
# plt.show()




# # make all likely wrong value of height and weight to nan  #
# df.loc[(df['Height (Inches)']<65)|(75<df['Height (Inches)']),'Height (Inches)']=np.nan
# df.loc[(df['Weight (Pounds)']<90)|(150<df['Weight (Pounds)']),'Weight (Pounds)']=np.nan

# # number of rows with nan #
# print(f'# of rows with nan : {df.isna().any(axis=1).sum()}\n')
# #number of nan for each col
# print(f'# of nan for each col\n{df.isna().sum()}\n')
# #all row without nan
# print(df.loc[df.isna().any(axis=1)==False,:])

# # impute missing value with SimpleImpute #
# df1=df.copy()
# sImp=impute.SimpleImputer(missing_values=np.nan, strategy='mean')
# before=df1[(df1['Height (Inches)'].isna())|(df1['Weight (Pounds)'].isna())|(df1['BMI'].isna())]
# after=before
# print(f'\nbefore simple impute\n{before}')
# indices=df1.index[(df1['Height (Inches)'].isna())|(df1['Weight (Pounds)'].isna())|(df1['BMI'].isna())].tolist()
# df1.loc[:,['Height (Inches)','Weight (Pounds)','BMI']]=sImp.fit_transform(df1.loc[:,['Height (Inches)','Weight (Pounds)','BMI']])
# for idx in indices:
#     after.loc[idx]=df1.loc[idx]
# print(f'\nafter simple impute\n{after[["Height (Inches)","Weight (Pounds)","BMI"]]}\n')




# impute missing value with Linear Regression #

# # remove missing data for Linear Regresion #
# from sklearn import linear_model
# reg=linear_model.LinearRegression()

# # Use original regression equation E #
# df2=df.copy()
# df2.loc[(df2['Height (Inches)']<55)|(75<df2['Height (Inches)']),'Height (Inches)']=np.nan
# df2.loc[(df2['Weight (Pounds)']<90)|(160<df2['Weight (Pounds)']),'Weight (Pounds)']=np.nan
# dIdx2=df2.index[(df2['Weight (Pounds)'].isna())|(df2['Height (Inches)'].isna())].tolist()
# # used for independant variables #
# learn=df2.dropna(how='any')

# # dependant variables #
# learn_hei=learn['Height (Inches)']
# learn_wei=learn['Weight (Pounds)']

# # NaN Height not NaN Weight #
# noHei=df2.loc[(df2['Height (Inches)'].isna()) & (df2['Weight (Pounds)'].notna())]
# noHei=noHei['Weight (Pounds)']
# # NaN Weight not NaN Height #
# noWei=df2.loc[(df2['Height (Inches)'].notna()) & (df2['Weight (Pounds)'].isna())]
# noWei=noWei[['Height (Inches)']]

# # learn model to predict weight using height #
# reg.fit(learn[['Height (Inches)']],learn_wei)
# #if missing values in height exist
# if(len(noHei)>0):
#     df2.loc[(df2['Height (Inches)'].isna()) & (df2['Weight (Pounds)'].notna()), 'Height (Inches)']=(noHei-reg.intercept_)/reg.coef_
# #if missing values in weight exist
# if(len(noWei)>0):
#     df2.loc[(df2['Height (Inches)'].notna()) & (df2['Weight (Pounds)'].isna()), 'Weight (Pounds)']=reg.predict(noWei)
# print(f'impute with equation E\n{df2.iloc[dIdx2]}\n\n')

# # draw scatter plot (height,weight) #
# px=df2['Height (Inches)'].to_numpy()
# py=df2['Weight (Pounds)'].to_numpy()
# dirty=df2.loc[dIdx2]
# dx=dirty['Height (Inches)'].to_numpy()
# dy=dirty['Weight (Pounds)'].to_numpy()
# plt.scatter(px,py)
# plt.scatter(dx,dy,color='r')
# incline=(dy.max()-dy.min())/(dx.max()-dx.min())
# plt.plot([dx.min()-2,dx.max()+3],[dy.min()-2*incline,dy.max()+3*incline], color='r')
# plt.title('Using Llinear Regression')
# plt.xlabel('Height (Inches)')
# plt.ylabel('Weight (Pounds)')
# plt.show()




# # Use another regression equation 1 #
# # Divided by female and male #
# df3=df.copy()
# df3.loc[(df3['Height (Inches)']<55) | (75<df3['Height (Inches)']),'Height (Inches)']=np.nan
# df3.loc[(df3['Weight (Pounds)']<90) | (160<df3['Weight (Pounds)']),'Weight (Pounds)']=np.nan

# # index for missed Weight value or missed Height value
# dIdx3=df3.index[(df3['Weight (Pounds)'].isna())|(df3['Height (Inches)'].isna())].tolist()
# # remove sample which has missed feature to learn model
# learn=df3.dropna(how='any')

# # Divided by female #

# # remove sample which has missed feature to learn model
# female=learn[learn['Sex']=='Female']
# # make dataframe which has none height and none weight
# noHei_female=df3[(df3['Sex']=='Female') & (df3['Height (Inches)'].isna()) & (df3['Weight (Pounds)'].notna())]
# noWei_female=df3[(df3['Sex']=='Female') & (df3['Height (Inches)'].notna()) & (df3['Weight (Pounds)'].isna())]
# if len(noWei_female)>0:
#     reg.fit(female[['Height (Inches)']],female['Weight (Pounds)'])
#     df3.loc[(df3['Sex']=='Female') & (df3['Height (Inches)'].notna()) & (df3['Weight (Pounds)'].isna()),'Weight (Pounds)']=reg.predict(noWei_female[['Height (Inches)']])
#     df3.loc[(df3['Sex']=='Female') & (df3['Height (Inches)'].isna()) & (df3['Weight (Pounds)'].notna()),'Height (Inches)']=(noHei_female['Weight (Pounds)']-reg.intercept_)/reg.coef_
# elif len(noHei_female)>0:
#     reg.fit(female[['Weight (Pounds)']],female['Height (Inches)'])
#     df3.loc[(df3['Sex']=='Female') & (df3['Height (Inches)'].isna()) & (df3['Weight (Pounds)'].notna()),'Height (Inches)']=reg.predict(noHei_female[['Weight (Pounds)']])
#     df3.loc[(df3['Sex']=='Female') & (df3['Height (Inches)'].notna()) & (df3['Weight (Pounds)'].isna()),'Weight (Pounds)']=(noWei_female['Height (Inches)']-reg.intercept_)/reg.coef_

# # Divided by male #

# # remove sample which has missed feature to learn model
# male=learn[learn['Sex']=='Male']
# # make dataframe which has none height and none weight
# noHei_male=df3[(df3['Sex']=='Male') & (df3['Height (Inches)'].isna()) & (df3['Weight (Pounds)'].notna())]
# noWei_male=df3[(df3['Sex']=='Male') & (df3['Height (Inches)'].notna()) & (df3['Weight (Pounds)'].isna())]
# if len(noWei_male)>0:
#     reg.fit(male[['Height (Inches)']],male['Weight (Pounds)'])
#     df3.loc[(df3['Sex']=='Male') & (df3['Height (Inches)'].notna()) & (df3['Weight (Pounds)'].isna()),'Weight (Pounds)']=reg.predict(noWei_male[['Height (Inches)']])
#     df3.loc[(df3['Sex']=='Male') & (df3['Height (Inches)'].isna()) & (df3['Weight (Pounds)'].notna()),'Height (Inches)']=(noHei_male['Weight (Pounds)']-reg.intercept_)/reg.coef_
# elif len(noHei_male)>0:
#     reg.fit(male[['Weight (Pounds)']],male['Height (Inches)'])
#     df3.loc[(df3['Sex']=='Male') & (df3['Height (Inches)'].isna()) & (df3['Weight (Pounds)'].notna()),'Height (Inches)']=reg.predict(noHei_male[['Weight (Pounds)']])
#     df3.loc[(df3['Sex']=='Male') & (df3['Height (Inches)'].notna()) & (df3['Weight (Pounds)'].isna()),'Weight (Pounds)']=(noWei_male['Height (Inches)']-reg.intercept_)/reg.coef_
# # print result
# print(f'impute with equation E(female&male)\n{df3.iloc[dIdx3]}\n\n')




# # Use another regression equation 2 #
# # Devided by BMI #
# df4=df.copy()
# df4.loc[(df4['Height (Inches)']<55) | (75<df4['Height (Inches)']),'Height (Inches)']=np.nan
# df4.loc[(df4['Weight (Pounds)']<90) | (160<df4['Weight (Pounds)']),'Weight (Pounds)']=np.nan

# # index for missed Weight value or missed Height value
# dIdx4=df4.index[(df4['Height (Inches)'].isna())|(df4['Weight (Pounds)'].isna())].tolist()

# # remove sample which has missed feature to learn model
# learn=df4.dropna(how='any')
# # make dataframe which has none height and none weight
# noHei=df4[(df4['Height (Inches)'].isna()) & (df4['Weight (Pounds)'].notna())]
# noWei=df4[(df4['Height (Inches)'].notna()) & (df4['Weight (Pounds)'].isna())]

# # impute weight and height for each BMI level group #
# # first, using height for independant variable and weight for dependant variable
# # if any sample doesn't have missed height, height is dependant variable and weight is independant variable

# #impute BMI level 1
# if(len(noWei[noWei['BMI']==0])>0):
#     reg.fit(learn.loc[learn['BMI']==0,['Height (Inches)']],learn.loc[learn['BMI']==0,'Weight (Pounds)'])
#     df4.loc[(df4['BMI']==0) & (df4['Height (Inches)'].notna()) & (df4['Weight (Pounds)'].isna()),'Weight (Pounds)']=reg.predict(noWei.loc[noWei['BMI']==0,['Height (Inches)']])
#     df4.loc[(df4['BMI']==0) & (df4['Height (Inches)'].isna()) & (df4['Weight (Pounds)'].notna()),'Height (Inches)']=(noHei.loc[noHei['BMI']==0,'Weight (Pounds)']-reg.intercept_)/reg.coef_
# elif(len(noHei[noHei['BMI']==0])>0):
#     reg.fit(learn.loc[learn['BMI']==0,['Weight (Pounds)']],learn.loc[learn['BMI']==0,'Height (Inches)'])
#     df4.loc[(df4['BMI']==0) & (df4['Height (Inches)'].notna()) & (df4['Weight (Pounds)'].isna()),'Weight (Pounds)']=(noWei.loc[noWei['BMI']==0,'Height (Inches)']-reg.intercept_)/reg.coef_
#     df4.loc[(df4['BMI']==0) & (df4['Height (Inches)'].isna()) & (df4['Weight (Pounds)'].notna()),'Height (Inches)']=reg.predict(noHei.loc[noHei['BMI']==0,['Weight (Pounds)']])
# #impute BMI level 2
# if(len(noWei[noWei['BMI']==1])>0):
#     reg.fit(learn.loc[learn['BMI']==1,['Height (Inches)']],learn.loc[learn['BMI']==1,'Weight (Pounds)'])
#     df4.loc[(df4['BMI']==1) & (df4['Height (Inches)'].notna()) & (df4['Weight (Pounds)'].isna()),'Weight (Pounds)']=reg.predict(noWei.loc[noWei['BMI']==1,['Height (Inches)']])
#     df4.loc[(df4['BMI']==1) & (df4['Height (Inches)'].isna()) & (df4['Weight (Pounds)'].notna()),'Height (Inches)']=(noHei.loc[noHei['BMI']==1,'Weight (Pounds)']-reg.intercept_)/reg.coef_
# elif(len(noHei[noHei['BMI']==1])>0):
#     reg.fit(learn.loc[learn['BMI']==1,['Weight (Pounds)']],learn.loc[learn['BMI']==1,'Height (Inches)'])
#     df4.loc[(df4['BMI']==1) & (df4['Height (Inches)'].notna()) & (df4['Weight (Pounds)'].isna()),'Weight (Pounds)']=(noWei.loc[noWei['BMI']==1,'Height (Inches)']-reg.intercept_)/reg.coef_
#     df4.loc[(df4['BMI']==1) & (df4['Height (Inches)'].isna()) & (df4['Weight (Pounds)'].notna()),'Height (Inches)']=reg.predict(noHei.loc[noHei['BMI']==1,['Weight (Pounds)']])
# #impute BMI level 3
# if(len(noWei[noWei['BMI']==2])>0):
#     reg.fit(learn.loc[learn['BMI']==2,['Height (Inches)']],learn.loc[learn['BMI']==2,'Weight (Pounds)'])
#     df4.loc[(df4['BMI']==2) & (df4['Height (Inches)'].notna()) & (df4['Weight (Pounds)'].isna()),'Weight (Pounds)']=reg.predict(noWei.loc[noWei['BMI']==2,['Height (Inches)']])
#     df4.loc[(df4['BMI']==2) & (df4['Height (Inches)'].isna()) & (df4['Weight (Pounds)'].notna()),'Height (Inches)']=(noHei.loc[noHei['BMI']==2,'Weight (Pounds)']-reg.intercept_)/reg.coef_
# elif(len(noHei[noHei['BMI']==2])>0):
#     reg.fit(learn.loc[learn['BMI']==2,['Weight (Pounds)']],learn.loc[learn['BMI']==2,'Height (Inches)'])
#     df4.loc[(df4['BMI']==2) & (df4['Height (Inches)'].notna()) & (df4['Weight (Pounds)'].isna()),'Weight (Pounds)']=(noWei.loc[noWei['BMI']==2,'Height (Inches)']-reg.intercept_)/reg.coef_
#     df4.loc[(df4['BMI']==2) & (df4['Height (Inches)'].isna()) & (df4['Weight (Pounds)'].notna()),'Height (Inches)']=reg.predict(noHei.loc[noHei['BMI']==2,['Weight (Pounds)']])
# #impute BMI level 4
# if(len(noWei[noWei['BMI']==3])>0):
#     reg.fit(learn.loc[learn['BMI']==3,['Height (Inches)']],learn.loc[learn['BMI']==3,'Weight (Pounds)'])
#     df4.loc[(df4['BMI']==3) & (df4['Height (Inches)'].notna()) & (df4['Weight (Pounds)'].isna()),'Weight (Pounds)']=reg.predict(noWei.loc[noWei['BMI']==3,['Height (Inches)']])
#     df4.loc[(df4['BMI']==3) & (df4['Height (Inches)'].isna()) & (df4['Weight (Pounds)'].notna()),'Height (Inches)']=(noHei.loc[noHei['BMI']==3,'Weight (Pounds)']-reg.intercept_)/reg.coef_
# elif(len(noHei[noHei['BMI']==3])>0):
#     reg.fit(learn.loc[learn['BMI']==3,['Weight (Pounds)']],learn.loc[learn['BMI']==3,'Height (Inches)'])
#     df4.loc[(df4['BMI']==3) & (df4['Height (Inches)'].notna()) & (df4['Weight (Pounds)'].isna()),'Weight (Pounds)']=(noWei.loc[noWei['BMI']==3,'Height (Inches)']-reg.intercept_)/reg.coef_
#     df4.loc[(df4['BMI']==3) & (df4['Height (Inches)'].isna()) & (df4['Weight (Pounds)'].notna()),'Height (Inches)']=reg.predict(noHei.loc[noHei['BMI']==3,['Weight (Pounds)']])
# #impute BMI level 5
# if(len(noWei[noWei['BMI']==4])>0):
#     reg.fit(learn.loc[learn['BMI']==4,['Height (Inches)']],learn.loc[learn['BMI']==4,'Weight (Pounds)'])
#     df4.loc[(df4['BMI']==4) & (df4['Height (Inches)'].notna()) & (df4['Weight (Pounds)'].isna()),'Weight (Pounds)']=reg.predict(noWei.loc[noWei['BMI']==4,['Height (Inches)']])
#     df4.loc[(df4['BMI']==4) & (df4['Height (Inches)'].isna()) & (df4['Weight (Pounds)'].notna()),'Height (Inches)']=(noHei.loc[noHei['BMI']==4,'Weight (Pounds)']-reg.intercept_)/reg.coef_
# elif(len(noHei[noHei['BMI']==4])>0):
#     reg.fit(learn.loc[learn['BMI']==4,['Weight (Pounds)']],learn.loc[learn['BMI']==4,'Height (Inches)'])
#     df4.loc[(df4['BMI']==4) & (df4['Height (Inches)'].notna()) & (df4['Weight (Pounds)'].isna()),'Weight (Pounds)']=(noWei.loc[noWei['BMI']==4,'Height (Inches)']-reg.intercept_)/reg.coef_
#     df4.loc[(df4['BMI']==4) & (df4['Height (Inches)'].isna()) & (df4['Weight (Pounds)'].notna()),'Height (Inches)']=reg.predict(noHei.loc[noHei['BMI']==4,['Weight (Pounds)']])
# # print result #
# print(f'impute with equation E(BMI)\n{df4.iloc[dIdx4]}\n\n')