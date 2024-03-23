# model_testing.py

import os
import pandas as pd
from sklearn.metrics import mean_squared_error
from model_preparation import trained_model

def load_data(test_folder):
    data = []
    for file in os.listdir(test_folder):
        if file.endswith(".csv"):
            file_path = os.path.join(test_folder, file)
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            data.append(df)
    test_data = pd.concat(data, ignore_index=True)
    return test_data

def model_testing(test_folder, model):
    test_data = load_data(test_folder)

    X_test = test_data.drop(columns=['temperature', 'date'])
    y_test = test_data['temperature']

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error on test data: {mse}")

test_folder = "test"
model_testing(test_folder, trained_model)
