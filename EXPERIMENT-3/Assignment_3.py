import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

data = {
    "Age": [20, 21, 22, 23, 24],
    "Salary": [25000, 30000, 28000, 35000, 40000],
    "Years of Experience": [1, 2, 3, 4, 5]
}

df = pd.DataFrame(data)

X = df[["Age", "Salary", "Years of Experience"]]

standard = StandardScaler()
standard_data = standard.fit_transform(X)

minmax = MinMaxScaler()
minmax_data = minmax.fit_transform(X)

print("--- StandardScaler ---")
print(standard_data)

print("\n--- MinMaxScaler ---")
print(minmax_data)

print("\nStandardScaler Range:")
print(standard_data.min(), "to", standard_data.max())

print("\nMinMaxScaler Range:")
print(minmax_data.min(), "to", minmax_data.max())