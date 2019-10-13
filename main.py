import pandas as pd
import math
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split


def data_pre_processing(train, predict):
    data = pd.concat([train, predict])

    dataFrame = data.drop(columns=['Hair Color'])

    dataFrame['Year of Record'].fillna(round(dataFrame['Year of Record'].median()), inplace=True)

    # Column Gender Manipulations
    dataFrame['Gender'].fillna('unknown', inplace=True)
    dataFrame['Gender'].replace(['0'], ['male'], inplace=True)
    dataFrame['Gender'] = dataFrame['Gender'].str.lower()
    dataFrame['Gender'] = dataFrame['Gender'].str.strip()

    # Column Age Manipulations
    dataFrame['Age'].fillna(round(dataFrame['Age'].median()), inplace=True)

    # Column Country Manipulation
    dataFrame['Country'] = dataFrame['Country'].str.lower()
    dataFrame['Country'] = dataFrame['Country'].str.strip()

    # Column Profession Manipulation
    dataFrame['Profession'].fillna('none', inplace=True)
    dataFrame['Profession'] = dataFrame['Profession'].str.lower()
    dataFrame['Profession'] = dataFrame['Profession'].str.strip()
    dataFrame['Profession'] = dataFrame['Profession'].str.slice(start=0, stop=3)
    dataFrame['Profession'] = dataFrame['Profession'].str.strip()

    # Column University Degree
    dataFrame['University Degree'].fillna('unknown', inplace=True)
    dataFrame['University Degree'].replace(['0'], ['No'], inplace=True)
    dataFrame['University Degree'] = dataFrame['University Degree'].str.lower()
    dataFrame['University Degree'] = dataFrame['University Degree'].str.strip()

    # cat codes/target mean encoding
    dataFrame['Gender'] = dataFrame['Gender'].astype('category').cat.codes
    dataFrame['Country'] = dataFrame['Country'].map(dataFrame.groupby('Country')['Income'].mean())
    dataFrame['Profession'] = dataFrame['Profession'].map(dataFrame.groupby('Profession')['Income'].mean())
    dataFrame['University Degree'] = dataFrame['University Degree'].map(dataFrame.groupby('University Degree')['Income'].mean())

    return dataFrame


def remove_outliers(data):
    data.rename(columns={'Income in EUR': 'Income'}, inplace=True)
    data.drop(data[data['Income'] < 0].index, inplace=True)
    data.drop(data['Income'].idxmax(), inplace=True)
    data.drop_duplicates(inplace=True, keep=False)

    return data


def split_dataset(data):
    mask = data['Income'] > 0
    trainDataFrame = data[mask]
    predictionDataFrame = data[~mask]

    return trainDataFrame, predictionDataFrame


def generate_csv(dataset, prediction):
    dataset['Income'] = prediction
    datasetToCSV = dataset[['Instance', 'Income']]
    datasetToCSV.to_csv('submission.csv', index=False)

    return datasetToCSV


def model_linear_regression_predict(X, y, predictionDataFrame):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

    linear_regression = LinearRegression()
    linear_regression.fit(X_train, y_train)

    y_pred = linear_regression.predict(X_test)
    print('Internal Test: Variance score: %.2f' % r2_score(y_test, y_pred))
    print("Internal Test: Root Mean squared error: %.2f" % math.sqrt(mean_squared_error(y_test, y_pred)))

    # predicting income on prediction dataset based on trained model
    predict(linear_regression, predictionDataFrame)


def model_random_forest_predict(X, y, predictionDataFrame):
    train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size=0.1, random_state=42)

    random_forest = RandomForestRegressor(n_estimators=200, random_state=42)
    random_forest.fit(train_features, train_labels)

    predictions = random_forest.predict(test_features)
    errors = abs(predictions - test_labels)
    print('Internal Test: Mean Absolute Error:', round(np.mean(errors), 2))
    mape = 100 * (errors / test_labels)
    accuracy = 100 - np.mean(mape)
    print('Internal Test: Accuracy:', round(accuracy, 2), '%.')
    print("Internal Test: Root Mean squared error: %.2f" % math.sqrt(mean_squared_error(test_labels, predictions)))

    # predicting income on prediction dataset based on trained model
    predict(random_forest, predictionDataFrame)


def predict(model, predictionDataFrame):
    P_test = predictionDataFrame.loc[:, predictionDataFrame.columns != 'Income']
    P_pred = model.predict(P_test)
    generate_csv(predictionDataFrame, P_pred)
    print('Predicted CSV generated')
