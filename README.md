# Machine Learning Model to Predict Income

#### Required dependencies: 
```
pandas, numpy, sklearn
```

#### Steps to Predict Output

1. In file "predict.py", uncomment the model that you want to use for prediction
2. Run the command
```
python predict.py
```

#### Project Flow
```
1. Reads the dataset provided to train the model (train.csv)
2. Removes the outliers from the training dataset
3. Reads the dataset on which income output is to be predicted (test.csv)
4. Merge both the set set and pre-process data (remove outlier/target mean encoding)
5. Split the dataset into train and test dataset
6. Performs training of the model on the training dataset
7. Returns CSV for the prediction of income on the test dataset
```