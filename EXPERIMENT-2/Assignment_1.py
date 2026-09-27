import pandas as pd
from sklearn.datasets import load_wine

wine=load_wine()

df=pd.DataFrame(wine.data,columns=wine.feature_names)

print("---First 5 rows:---")
print(df.head())

print("\n---shape----")
print(df.shape)

print("\n-----Dataset Information:-----")
print(df.info())

print("\n---Statistical summary:---")
print(df.describe())

print("\n----missing values:----")
print(df.isnull().sum())

print("\n---Data types:---")
print(df.dtypes)

print("\n---correlation matrix---")
print(df.corr(numeric_only=True))





