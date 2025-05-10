import numpy as np
import pandas as pd

grade=pd.Series(['A','B','B','A','A','C'])
df=pd.DataFrame({'id':[2,3,5,6,7,9],
                 'grade':grade.astype('category')})

#grade 컬럼에 몇개의 카테고리가 있는지 알려줌
print(df['grade'])
#cat: 카테고리에 접근
#catgories: 카테고리의 metadata 알려줌
#set_categories:categories를 재구성할 수 있음
print(df['grade'].cat.categories)
#'grade' series의 카테고리를 수정
#기존 grade series의 A,B,C는 더이상 유효하지 않음 -> nan으로 변경됨
#rename_categories 사용하면 A,B,C의 명칭을 변경할 수 있음
df['grade']=df['grade'].cat.set_categories(['수','우','미','양','가'])

#missing data#
df2=df.copy()
df2['failed']=df['grade']==pd.Series(['C']*6)
#idx=3,4,5 row의 class에는 NaN 저장됨
df2['class']=pd.Series(['high','high','mid'])
#nan이 존재하는 row를 제거
# df2.dropna(how='any')
# #nan을 value로 채움
# df2.fillna(value=0)
# #nan이 있는 cell은 true, 없으면 false
# df2.isna()

#FILE_I/O#
#CSV파일
#현재 파이썬 폴더를 csv파일 형태로 아래 경로에 저장
#경로를 지정하지 않으면 현재 파이썬 폴더가 위치한 경로에 저장
df.to_csv("c:/Gachon/25_1/DS/df.csv")
#csv파일에 접근해서 dataFrame을 읽을 수 있음
#index_col으로 index 역할을 하는 컬럼을 알려줄 수 있음
df3=pd.read_csv("c:/Gachon/25_1/DS/df.csv",index_col=0)
# df3=pd.read_csv("c:/Gachon/25_1/DS/df.csv",index_col='COL_NAME')

#EXCEL파일
df.to_excel('c:/Gachon/25_1/DS/df.xlsx', sheet_name='Sheet1')
df4=pd.read_excel('c:/Gachon/25_1/DS/df.xlsx',index_col=None)
print(df4)