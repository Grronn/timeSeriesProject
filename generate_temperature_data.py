import os
import random
import pandas as pd

def generate_temperature_data(start_date, end_date, anomalies_probability, noise_level):
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    temperature_data = []

    previous_temperature = 0  

    for date in date_range:
        temperature = 0
        pressure = 0
        wind_speed = 0
        humidity = 0
        precipitation = 0

        if start_date.month == 1:  # Январь
            temperature = random.uniform(-35, -15)
        elif start_date.month == 7:  # Июль
            temperature = random.uniform(15, 40)
        elif start_date.month == 9:  # Сентябрь
            temperature = random.uniform(0, 15)

        if random.uniform(0, 1) < anomalies_probability:
            temperature += random.uniform(-5, 5)

        temperature = min(max(previous_temperature - 5, temperature), previous_temperature + 5)

        temperature += random.uniform(-noise_level, noise_level)

        if temperature < 0:
            pressure = random.uniform(970, 1010) + random.uniform(-5, 5)
        elif temperature >= 0 and temperature < 20:
            pressure = random.uniform(980, 1020) + random.uniform(-5, 5)
        else:
            pressure = random.uniform(990, 1030) + random.uniform(-5, 5)

        wind_speed = random.uniform(0, 30) + random.uniform(-5, 5)

        humidity = random.uniform(0, 100) + random.uniform(-10, 10)

        precipitation = random.uniform(0, 50) + random.uniform(-10, 10)

        temperature = round(temperature)
        pressure = round(pressure)
        wind_speed = round(wind_speed, 1)
        humidity = round(humidity, 1)
        precipitation = round(precipitation, 1)

        temperature_data.append({'date': date, 'temperature': temperature, 'pressure': pressure,
                                 'wind_speed': wind_speed, 'humidity': humidity, 'precipitation': precipitation})

        previous_temperature = temperature  

    return temperature_data

def save_data_to_file(data, folder, filename):
    file_path = os.path.join(folder, filename)
    data_frame = pd.DataFrame(data)
    data_frame.to_csv(file_path, index=False)

datasets_parameters = [
    {'start_date': pd.to_datetime('2024-01-01'), 'end_date': pd.to_datetime('2024-01-31'), 'anomalies_probability': 0.1, 'noise_level': 2},
    {'start_date': pd.to_datetime('2024-07-01'), 'end_date': pd.to_datetime('2024-07-31'), 'anomalies_probability': 0.15, 'noise_level': 3},
    {'start_date': pd.to_datetime('2024-09-01'), 'end_date': pd.to_datetime('2024-09-30'), 'anomalies_probability': 0.2, 'noise_level': 2.5},
    {'start_date': pd.to_datetime('2025-01-01'), 'end_date': pd.to_datetime('2025-01-31'), 'anomalies_probability': 0.1, 'noise_level': 2},
    {'start_date': pd.to_datetime('2025-07-01'), 'end_date': pd.to_datetime('2025-07-31'), 'anomalies_probability': 0.15, 'noise_level': 3},
    {'start_date': pd.to_datetime('2025-09-01'),'end_date': pd.to_datetime('2025-09-30'), 'anomalies_probability': 0.2, 'noise_level': 2.5}
]

for i, params in enumerate(datasets_parameters, start=1):
    train_data = generate_temperature_data(**params)
    test_data = generate_temperature_data(**params)

    os.makedirs('train', exist_ok=True)
    os.makedirs('test', exist_ok=True)

    save_data_to_file(train_data, 'train', f'train_data_set_{i}.csv')
    save_data_to_file(test_data, 'test', f'test_data_set_{i}.csv')
