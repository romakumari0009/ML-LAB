import pandas as pd

data={
       "Student Name":["Amit","Raj","Sonali","piyu","Neeti"],
       "Roll number":[101,102,103,104,105],
       "Marks":[89,45,90,67,78],
       "Attendance":[90,89,80,45,78]
     }
df=pd.DataFrame(data)

def grade(marks):
        if(marks>=90):
                return "A"
        elif(marks>=80):
                return "B"
        elif(marks>=70):
                return "C"
        elif(marks>=60):
                return "D"
        else:
               return "F"
df["Grade"]=df["Marks"].apply(grade)

print(df)





