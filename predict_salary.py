# A simple app to predict the salary based on the number of years of experience
# Import all the required lib
#joblib : library 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib

#Loading the dataset
df = pd.read_csv("salary_data.csv")
#print(df.info())

#Split the data into Target Variable and independent Variables
X = df[["YearsExperience"]]
Y = df[["Salary"]]

#Train-test-split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=1)

# Sacling down the data 
# crating an object of standard module 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

#train the model
model=LinearRegression()
model.fit(X_train_scaled,Y_train)

#save the model and Scaler
joblib.dump(model,"predict_salary.pkl")
joblib.dump(scaler,"scaler.pkl")
print("Model and Scaler are saved")


