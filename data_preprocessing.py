import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_file, output_file):
    # Загрузка данных
    data = pd.read_csv(input_file)

    # Преобразование столбца с датами в формат datetime
    data['date'] = pd.to_datetime(data['date'])

    # Добавление столбца с месяцем
    data['month'] = data['date'].dt.month

    # Разделение данных на признаки и целевую переменную
    X = data.drop(columns=['date', 'temperature'])
    y = data['temperature']

    # Применение StandardScaler для стандартизации признаков
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Сохранение предобработанных данных
    processed_data = pd.DataFrame(X_scaled, columns=X.columns)
    processed_data['temperature'] = y  # Добавление целевой переменной обратно
    processed_data.to_csv(output_file, index=False)

    return processed_data

if __name__ == "__main__":
    input_file = "train/train_data_set_1.csv"  # Путь к входному файлу
    output_file = "train/processed_train_data_set_1.csv"  # Путь к файлу с предобработанными данными
    processed_data = preprocess_data(input_file, output_file)

    # Для демонстрации просто выведем первые строки предобработанных данных
    print(processed_data.head())