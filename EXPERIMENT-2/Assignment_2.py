import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
import pandas as pd

wine=load_wine()
df=pd.DataFrame(wine.data,columns=wine.feature_names)

plt.figure(figsize=(15,10))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.title("Boxplots of wine dataset features")
plt.savefig("A2_fig")
plt.show()