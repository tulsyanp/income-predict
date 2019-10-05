import pandas as pd


def data_pre_processing(train, predict):

    data = pd.concat([train, predict])

    dataFrame = data.drop(columns=['Wears Glasses', 'Hair Color', 'Body Height [cm]', 'Size of City'])

    # TODO Column Year of Record (Mode/Mean)
    dataFrame['Year of Record'].fillna(dataFrame['Year of Record'].mode()[0], inplace=True)

    # Column Gender Manipulations
    dataFrame['Gender'].fillna('unknown', inplace=True)
    dataFrame['Gender'].replace(['0'], ['male'], inplace=True)
    dataFrame['Gender'] = dataFrame['Gender'].str.lower()

    # Column Age Manipulations
    dataFrame['Age'].fillna(round(dataFrame['Age'].mean()), inplace=True)

    # Column Country Manipulation
    dataFrame['Country'] = dataFrame['Country'].str.lower()

    # Column Profession Manipulation
    dataFrame['Profession'].fillna('none', inplace=True)
    dataFrame['Profession'] = dataFrame['Profession'].str.lower()

    # Column University Degree
    dataFrame['University Degree'].fillna('unknown', inplace=True)
    dataFrame['University Degree'].replace(['0'], ['No'], inplace=True)
    dataFrame['University Degree'] = dataFrame['University Degree'].str.lower()

    # encoding
    dataFrame = pd.get_dummies(
        dataFrame,
        columns=["Gender", "Country", "Profession", "University Degree"],
        prefix=["Gender", "Country", "Profession", "Degree"]
    )

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
