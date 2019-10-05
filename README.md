# Machine Learning Model to Predict Income

#### Required dependencies: 
```
pandas, sklearn
```

#### Command to Predict Output
```
python lr_predict.py
```

#### Project Flow
```
1. Reads the dataset provided to train the model (train.py)
2. Removes the outliers from the training dataset
3. Reads the dataset on which income output is to be predicted (test.py)
4. Merge both the set set and pre-process data (remove outlier/one hot encoding)
5. Split the dataset into train and test dataset
6. Performs training of the model on the training dataset
7. Returns CSV for the prediction of income on the test dataset
```