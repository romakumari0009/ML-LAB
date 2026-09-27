import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

wine=load_wine()
df=pd.DataFrame(wine.data,columns=wine.feature_names)
correlation=df.corr()

plt.figure(figsize=(12,8))
sns.heatmap(correlation,annot=True,cmap="coolwarm",fmt=".2f")

plt.title("Correlation heatmap of wine Dataset")

plt.savefig("A3_fig")
plt.show()

print("Strongest positive correlation:")
print(correlation.stack().sort_values(ascending=False).iloc[1])

