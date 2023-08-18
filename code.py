#Python file for Rock Vs Mine Prediction using Machine Learning

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#loading the dataset to a pandas Dataframe
sonar_data = pd.read_csv('/content/Sonar_Data.csv', header = None)

#to display the 1st five items
sonar_data.head()

#to see the number of rows and columns
sonar_data.shape

#to see the measures of central tendency
sonar_data.describe()

#for the count of number of rocks and mines
# M for mine
# R for rock
sonar_data[60].value_counts()
sonar_data.groupby(60).mean()

#separating data and labels
x = sonar_data.drop(columns = 60, axis = 1)
y = sonar_data[60]

print(x)
print(y)

#splitting into training and testing data
# test_size specifies the percentage of data that is to be trained
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, stratify = y, random_state = 1)

print(x.shape, x_train.shape, x_test.shape)
print(x_train)
print(y_train)

model = LogisticRegression()

#training the LR model with training data
model.fit(x_train, y_train)

#accuracy of the training data
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)

print('Accuracy of training data : ', training_data_accuracy)

#accuracy of test data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)

print('Accuracy of test data : ', test_data_accuracy)

input_data = (0.0261,0.0266,0.0223,0.0749,0.1364,0.1513,0.1316,0.1654,0.1864,0.2013,0.2890,0.3650,0.3510,0.3495,0.4325,0.5398,0.6237,0.6876,0.7329,0.8107,0.8396,0.8632,0.8747,0.9607,0.9716,0.9121,0.8576,0.8798,0.7720,0.5711,0.4264,0.2860,0.3114,0.2066,0.1165,0.0185,0.1302,0.2480,0.1637,0.1103,0.2144,0.2033,0.1887,0.1370,0.1376,0.0307,0.0373,0.0606,0.0399,0.0169,0.0135,0.0222,0.0175,0.0127,0.0022,0.0124,0.0054,0.0021,0.0028,0.0023)

#changing the input datatype to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the np array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if(prediction[0] == 'R'):
  print('The object is a Rock.')
else:
  print('The object is a Mine!!!')
