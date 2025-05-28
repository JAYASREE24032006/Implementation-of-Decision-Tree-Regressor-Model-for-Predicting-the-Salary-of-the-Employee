# EX-9 : Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee
### Name : R.Jayasree
### R.No : 212223040074

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the libraries and read the data frame using pandas.

2.Calculate the null values present in the dataset and apply label encoder. 

3.Determine test and training data set and apply decison tree regression in dataset. 

4.Calculate Mean square error,data prediction and r2.
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: R.JAYASREE
RegisterNumber:  212223040074
*/
```
```

import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])



```

## Output:
### DATA HEAD


![image](https://github.com/user-attachments/assets/cb3ebf5d-150e-4887-896b-2e61cb8df184)

### DATA INFO 
![image](https://github.com/user-attachments/assets/50781911-e4d0-4376-a95e-b9557d4cc8af)

### DATA HEAD FOR SALARY

![image](https://github.com/user-attachments/assets/80892400-f828-4323-b357-cc51b8895af4)

### MEAN SQUARED ERROR
![image](https://github.com/user-attachments/assets/3453a2a2-1b9c-4903-be5a-4dc1858f18f3)

### R VALUE 
![image](https://github.com/user-attachments/assets/283a65f9-b4ab-447f-87d3-21a94220eb96)

### DATA PREDICTION

![image](https://github.com/user-attachments/assets/3398eff9-c7ed-4dce-933d-5568dd1fa32e)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
