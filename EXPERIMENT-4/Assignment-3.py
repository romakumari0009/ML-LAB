import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv("Housing.csv.xls")
df = df.dropna(subset=["area", "price"])

X = df[["area"]]
y = df["price"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)
linear_r2 = r2_score(y_test, linear_pred)


poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

poly_pred = poly_model.predict(X_test_poly)
poly_r2 = r2_score(y_test, poly_pred)


print("Linear Regression R2 Score:", round(linear_r2, 4))
print("Polynomial Regression R2 Score:", round(poly_r2, 4))


if poly_r2 > linear_r2:
    print("Polynomial Regression has a higher R2 score.")
elif poly_r2 < linear_r2:
    print("Linear Regression has a higher R2 score.")
else:
    print("Both models have the same R2 score.")


plt.figure(figsize=(8, 5))

plt.scatter(X, y, label="Actual Data")


X_plot = np.sort(X.values, axis=0)
X_plot_poly = poly.transform(X_plot)

plt.plot(
    X_plot,
    linear_model.predict(X_plot),
    linewidth=2,
    label="Linear Regression"
)

plt.plot(
    X_plot,
    poly_model.predict(X_plot_poly),
    linewidth=2,
    label="Polynomial Regression"
)

plt.xlabel("House Area (sq. ft)")
plt.ylabel("House Price")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.grid(True)

plt.savefig("Exp4_A3.png")
plt.show()