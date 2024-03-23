#!/bin/bash

# Запуск скрипта для создания и сохранения данных
python generate_temperature_data.py

# Запуск скрипта для предобработки данных
python data_preprocessing.py

# Запуск скрипта для подготовки и обучения модели
python model_preparation.py

# Запуск скрипта для тестирования модели на тестовых данных
python model_testing.py