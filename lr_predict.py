import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from main import data_pre_processing, remove_outliers, split_dataset, generate_csv
import math


# read training dataset
trainingData = pd.read_csv("train.csv")
trainData = remove_outliers(trainingData)

# read prediction dataset
predictionData = pd.read_csv("test.csv")

# data pre processing
preProcessedDataFrame = data_pre_processing(trainData, predictionData)

# split dataset
trainDataFrame, predictionDataFrame = split_dataset(preProcessedDataFrame)

# model training by dividing target and attributes
X = trainDataFrame.loc[:, trainDataFrame.columns != 'Income']
y = trainDataFrame['Income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)

y_pred = linear_regression.predict(X_test)
print('Variance score: %.2f' % r2_score(y_test, y_pred))
print("Root Mean squared error: %.2f" % math.sqrt(mean_squared_error(y_test, y_pred)))


# predicting income on prediction dataset based on trained model
P_test = predictionDataFrame.loc[:, predictionDataFrame.columns != 'Income']
P_pred = linear_regression.predict(P_test)
pred_to_csv = generate_csv(predictionDataFrame, P_pred)
print('CSV generated')

