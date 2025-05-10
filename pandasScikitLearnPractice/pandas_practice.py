import numpy as np
import pandas as pd

# #Series#
# s=pd.Series([1,2,3,4,np.nan,6,7])
# print(s)

# # dataFrame#
# df=pd.DataFrame(np.random.randn(6,4))
# print(df)

#date_range, 날짜형식으로 반환#
#freq 설정 안하면 D 단위로 증가함
dates=pd.date_range('20250409',periods=7,freq='YE')
print(dates)

#dataFrame을 다른 dataFrame의 인덱스로 사용 가능#
#위에서 만든 dates를 df의 인덱스로 사용
#col의 인덱스에 알파벳 사용
df=pd.DataFrame(np.random.randn(7,4),index=dates,
                columns=['A','B','C','D'])
print(df)

#dictionary 구조를 이용한 dataFrame 만들기#
#key :value 구조
#key: col의 이름/인덱스 역할
df2=pd.DataFrame({'A':1.,
                  'B':pd.Timestamp('20250409'),
                  'C':pd.Series([1,2,3,4],dtype='float32'),
                  'D':np.array([3]*4,dtype='int32'),
                  'E':pd.Categorical(["train","test","train","test"]),
                  'F':'foo'})

#맨 위 또는 맨 아래 행을 확인 가능
print(f"head of df2\n {df2.head()}") #5개가 디폴트
print(f"tail of df2\n {df2.tail(2)}") #개수 지정 가능
print(f"index\n {df2.index}")
print(f"col\n {df2.columns}")
#numpy 형태로 반환가능
print(df2.to_numpy)
#B를 기준으로 정렬, 오름차순이 디폴트
print(f"{df2.sort_values(by='B', ascending=False)}")

#dataFrame selection#
#slicing으로 행 출력
print(df2[1:2])
#컬럼인덱스로 다중 열 출력
print(f"열1\n{df2[['A','B']]}\n\n")
print(df2.B)
#조건에 맞는 열 selection
print(df2.E=="test") #E에 대해 true or false 반환
print(df2[df2.E=="train"])

#label을 이용한 selection
#행 출력
print(df2.loc[[2,3]])
#열 출력
#다중 열 출력할 때는 리스트 형태로 입력해야 함.
#[c1,c2,c3,c4,...]
print(f"열2\n{df2.loc[:,'A']}\n\n")

#position을 이용한 selection
#row
print(df2.iloc[0])
#row,col
print(df2.iloc[1:3,1:5])
print(df2.iloc[[1,2],[1,2,3,4]])

#dataFrame에 열 추가
df3=df2.copy()
df3.F=pd.Series([1,2,3,4],index=(0,1,2,3))
print(df3)

#at, iat, loc
print("\ndf3\n")
df3.at[2,'A']=0
print(df3)
df3.iat[2,0]=1
print(df3)
df3.loc[0,'B']=pd.Timestamp('20250101')
print(df3)