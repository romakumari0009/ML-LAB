import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

#1.load dataset
iris=load_iris()
df=pd.DataFrame(iris.data,columns=iris.feature_names)
df["target"]=iris.target

#2.Basic data exploration
print("---First five rows---")
print(df.head())

print("\n---Dataset information---")
print(df.info())

print("\n---Statistical summary---")
print(df.describe())

print("\n---Missing values---")
print(df.isnull().sum())

#3.Correlation matrix
print("\n---Correlation Matrix---")
print(df.corr(numeric_only=True))

#4.Scatter plot:Sepal length vs Petal length
plt.figure(figsize=(7,5))
sns.scatterplot(
           data=df,
           x="sepal length (cm)",
           y="petal length (cm)",
           hue="target",
           palette="viridis"
)
plt.title("Sepal length vs Petal length by Class")
plt.savefig("Exp2_fig1")
plt.show()


#5.Histogram:Distribution of sepal length
plt.figure(figsize=(7,5))
sns.histplot(df["sepal length (cm)"],kde=True,color="blue")
plt.title("Distribution of  sepal length")
plt.savefig("Exp2_fig2")
plt.show()



