import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_file, output_file):
    data = pd.read_csv(input_file)

    data['date'] = pd.to_datetime(data['date'])

    data['month'] = data['date'].dt.month

    X = data.drop(columns=['date', 'temperature'])
    y = data['temperature']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    processed_data = pd.DataFrame(X_scaled, columns=X.columns)
    processed_data['temperature'] = y
    processed_data.to_csv(output_file, index=False)

    return processed_data

if __name__ == "__main__":
    input_file = "train/train_data_set_1.csv" 
    output_file = "train/processed_train_data_set_1.csv" 
    processed_data = preprocess_data(input_file, output_file)

    print(processed_data.head())
