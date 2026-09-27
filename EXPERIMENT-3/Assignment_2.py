import pandas as pd
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

data={"Age":[20,23,None,30,24],"Salary":[25000,30000,50000,None,35000],"Department":["IT","HR","IT","Finance","IT"],"Years of Experience":[1,3,2,None,5]}

df=pd.DataFrame(data)

X=df

numeric_features=["Age","Salary","Years of Experience"]

categorical_features=["Department"]

numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")),("scaler", MinMaxScaler())])
categorical_transformer=Pipeline(steps=[("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))])

preprocessor=ColumnTransformer(transformers=[("num", numeric_transformer, numeric_features),("cat", categorical_transformer, categorical_features)])

X_processed=preprocessor.fit_transform(X)

print("--- Processed Data using MinMaxScaler ---")

print(X_processed)      