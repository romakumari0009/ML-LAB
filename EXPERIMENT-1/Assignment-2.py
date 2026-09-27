import pandas as pd

data={
       "Student_Name":["abc","def","gih","ijk","lmn"],
       "Roll_Number":[101,102,103,104,105],
       "Marks":[67,45,90,89,99],
       "Attendance":[90,89,56,78,88]
     }
df=pd.DataFrame(data)
print("Complete DataFrame:")
print(df)

print("Students scoring above 80:")
print(df[df["Marks"]>80])