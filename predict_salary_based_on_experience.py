# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# importing datasets and extracting dependent and independent variables
salary_data = pd.read_csv(r'Salary_Data.csv')
x= salary_data.iloc[:, :-1].values
y= salary_data.iloc[:,1].values

# visualizing data
sns.distplot(salary_data['YearsExperience'],kde=False,bins=10)
sns.countplot(y='YearsExperience',data=salary_data)
sns.barplot(x='YearsExperience',y='Salary',data=salary_data)
sns.heatmap(salary_data.corr())

# splitting data into training set and test set
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size= 0.2,random_state=0)

# loading model
reg_model=LinearRegression()

# fitting linearRegression to train dataset
reg_model.fit(x_train,y_train)

# making prediction with test dataset
y_pred=reg_model.predict(x_test)

# visualizing the training dataset
plt.scatter(x_train,y_train,color='b')
plt.plot(x_train,reg_model.predict(x_train),c='r')
plt.title('Salary vrs experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# visualizing the testing dataset
plt.scatter(x_test,y_pred,c='b')
plt.plot(x_train,reg_model.predict(x_train),c='r')
plt.title('Salary vrs Experience (Testing Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Finding the residuals
print('MAE :',metrics.mean_absolute_error(y_test,y_pred))
print('MSE :',metrics.mean_squared_error(y_test,y_pred))
print('RMSE :',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))