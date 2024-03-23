# model_preparation.py

import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def load_data(train_folder):
    data = []
    processed_file = None
    for file in os.listdir(train_folder):
        if file.endswith(".csv"):
            file_path = os.path.join(train_folder, file)
            if "processed" in file:
                processed_file = file_path
            else:
                df = pd.read_csv(file_path)
                df['date'] = pd.to_datetime(df['date'])
                df['year'] = df['date'].dt.year
                df['month'] = df['date'].dt.month
                df['day'] = df['date'].dt.day
                data.append(df)
    train_data = pd.concat(data, ignore_index=True)
    return train_data, processed_file

def model_preparation(train_folder):
    train_data, processed_file = load_data(train_folder)

    if processed_file:
        processed_folder = os.path.join(train_folder, "processed_train")
        os.makedirs(processed_folder, exist_ok=True)
        shutil.move(processed_file, processed_folder)
        print(f"Processed file moved to: {processed_folder}")

    X = train_data.drop(columns=['temperature', 'date'])
    y = train_data['temperature']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    print(f"Mean Squared Error: {mse}")
    return model

trained_model = model_preparation("train")
