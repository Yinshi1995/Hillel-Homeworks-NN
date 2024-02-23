import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

dataset = pd.read_csv('your_dataset.csv')  

X = dataset.drop(columns=['target_column'])  
y = dataset['target_column']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model1 = LinearRegression()
model1.fit(X_train, y_train)

y_pred1 = model1.predict(X_test)

mse1 = mean_squared_error(y_test, y_pred1)
print("Втрати для першої моделі:", mse1)

model2 = LinearRegression()
model2.fit(X_train, y_train)

y_pred2 = model2.predict(X_test)

mse2 = mean_squared_error(y_test, y_pred2)
print("Втрати для другої моделі:", mse2)

theta_normal_eq = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)
y_pred_normal_eq = X_test.dot(theta_normal_eq)
mse_normal_eq = mean_squared_error(y_test, y_pred_normal_eq)
print("Втрати для моделі зі звичайним рівнянням:", mse_normal_eq)

plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred1, color='blue', linewidth=3)
plt.plot(X_test, y_pred2, color='red', linewidth=3)
plt.xlabel('Features')
plt.ylabel('Target')
plt.legend(['Model 1', 'Model 2', 'Actual'])
plt.show()
