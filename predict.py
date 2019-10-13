import pandas as pd
from main import data_pre_processing, remove_outliers, split_dataset, model_linear_regression_predict, model_random_forest_predict


# read training dataset
trainingData = pd.read_csv("train.csv")
trainData = remove_outliers(trainingData)

# read prediction dataset
predictionData = pd.read_csv("test.csv")

# data pre processing
preProcessedDataFrame = data_pre_processing(trainData, predictionData)

# split dataset
trainDataFrame, predictionDataFrame = split_dataset(preProcessedDataFrame)

# prediction dataset processing
predictionDataFrame['Country'] = predictionDataFrame['Country'].fillna(round(predictionDataFrame['Country'].mean()))
predictionDataFrame['Profession'] = predictionDataFrame['Profession'].fillna(round(predictionDataFrame['Profession'].mean()))

# model training by dividing target and attributes
X = trainDataFrame.loc[:, trainDataFrame.columns != 'Income']
y = trainDataFrame['Income']

# Uncomment to predict using Linear Regression Model
# model_linear_regression_predict(X, y, predictionDataFrame)

# Uncomment to predict using Random Forest Model
model_random_forest_predict(X, y, predictionDataFrame)
