import pandas as pd
import numpy as np

#sub4,sub5,sub6,sub9,sub11,sub15,sub18,sub20,sub22,sub23,sub27

df1=pd.read_csv("sub4.csv")
df2=pd.read_csv("sub5.csv")
df3=pd.read_csv("sub6.csv")
df4=pd.read_csv("sub9.csv")
df5=pd.read_csv("sub11.csv")
df6=pd.read_csv("sub15.csv")
df7=pd.read_csv("sub18.csv")
df8=pd.read_csv("sub20.csv")
df9=pd.read_csv("sub22.csv")
df10=pd.read_csv("sub23.csv")
df11=pd.read_csv("sub27.csv")

df_all=df1.append([df2,df3,df4,df5,df6,df7,df8,df9,df10,df11])
ensembled_ans=df_all.groupby('roadId',as_index=False)['noOfLanes'].agg(lambda x: x.value_counts().index[0])
ensembled_ans.to_csv("sub28.csv",index=False)


df_all=pd.read_csv("sub1.csv")
for i in range(2,38):
	df_all=df_all.append(pd.read_csv("sub"+str(i)+".csv"))

ensembled_ans=df_all.groupby('roadId',as_index=False)['noOfLanes'].agg(lambda x: x.value_counts().index[0])
ensembled_ans.to_csv("sub38.csv",index=False)



#18,36
df_all=pd.read_csv("sub6.csv")
for i in [18,36]:
	df_all=df_all.append(pd.read_csv("sub"+str(i)+".csv"))

ensembled_ans=df_all.groupby('roadId',as_index=False)['noOfLanes'].agg(lambda x: x.value_counts().index[0])
ensembled_ans.to_csv("sub39.csv",index=False)

