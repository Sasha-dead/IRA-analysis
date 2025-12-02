# data_collector.py - Модуль збору та попередньої обробки даних

import numpy as np
import pandas as pd
from skfuzzy import gaussmf
import paho.mqtt.client as mqtt
import requests
import logging

logging.basicConfig(level=logging.INFO)

def collect_data(source='mqtt', broker='localhost', topic='iot/data', url=None, file_path=None):
    """
    Збір даних з IoT-джерел.
    - source: 'mqtt' для реального часу, 'http' для API, 'file' для локальних файлів.
    """
    if source == 'mqtt':
        data = []  # Симуляція збору
        def on_message(client, userdata, msg):
            data.append(msg.payload.decode())
        client = mqtt.Client()
        client.on_message = on_message
        client.connect(broker)
        client.subscribe(topic)
        client.loop_start()
        # Симуляція: зібрати дані за 10 сек
        import time
        time.sleep(10)
        client.loop_stop()
        return pd.DataFrame({'timestamp': range(len(data)), 'value': data})  # Приклад
    elif source == 'http' and url:
        response = requests.get(url)
        if response.status_code == 200:
            return pd.DataFrame(response.json())
        else:
            logging.error("HTTP error")
            return None
    elif source == 'file' and file_path:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            return pd.read_json(file_path)
    logging.error("Invalid source")
    return None

def normalize_data(df):
    """Нормалізація даних Min-Max."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / (df[numeric_cols].max() - df[numeric_cols].min())
    return df

def fuzzify_data(series, c=0.5, sigma=0.2):
    """Fuzzification з гаусівською функцією."""
    return gaussmf(series, c, sigma)

def extract_features(df, window=10):
    features = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        rolling = df[col].rolling(window)
        features[f'{col}_mean'] = rolling.mean()
        features[f'{col}_var'] = rolling.var()
    return pd.DataFrame(features)


# data_collector.py

def preprocess(df):
    # --- 1. Обробка міток (Labels) ---
    label_col = None
    possible_labels = ["set[string]", "label", "tunnel_parents", "class"]

    # Шукаємо колонку з мітками
    for col in possible_labels:
        if col in df.columns:
            # Перевіряємо, чи схожа вона на мітку (містить 'Malicious' або 'Benign')
            if df[col].astype(str).str.contains('Malicious|Benign|Attack', case=False).any():
                label_col = col
                break

    if not label_col:
        # Якщо не знайшли, можливо це 'tunnel_parents' або інша (специфіка датасету)
        # Спробуємо знайти будь-яку текстову колонку в кінці
        if 'label' in df.columns:
            label_col = 'label'
        elif 'tunnel_parents' in df.columns:
            label_col = 'tunnel_parents'
        else:
            print(f"Увага: Колонку міток не знайдено. Стовпці: {list(df.columns)}")
            # Спробуємо створити фіктивну, якщо це тест, або викинути помилку
            df['label'] = 0
            label_col = 'label'

    # Створюємо y (1 = Malicious, 0 = Benign)
    y = df[label_col].apply(lambda x: 1 if "malicious" in str(x).lower() else 0)

    # --- 2. Видалення метаданих ---
    # Видаляємо IP адреси, порти, ID транзакцій, бо LSTM потрібні числові патерни поведінки
    cols_to_drop = [label_col, 'ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p', 'service', 'proto']
    df_numeric = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    # --- 3. Очищення даних (Ключовий момент) ---
    # Замінюємо '-' на NaN
    df_numeric = df_numeric.replace('-', np.nan)

    # Конвертуємо все в числа (помилки стають NaN)
    for col in df_numeric.columns:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')

    # !!! ВИПРАВЛЕННЯ: ЗАМІСТЬ ВИДАЛЕННЯ РЯДКІВ, ЗАПОВНЮЄМО НУЛЯМИ !!!
    # У трафіку NaN часто означає 0 байт або 0 тривалість
    df_numeric = df_numeric.fillna(0)

    # Видаляємо колонки, які містять ТІЛЬКИ нулі (вони не несуть інформації)
    df_numeric = df_numeric.loc[:, (df_numeric != 0).any(axis=0)]

    if df_numeric.empty:
        raise ValueError("Помилка: Після очищення не залишилося колонок з даними.")

    # --- 4. Додавання Fuzzy ознак ---
    df_clean = df_numeric.copy()

    # Вибираємо тільки колонки з варіативністю (std > 0) для fuzzy логіки
    valid_cols = []
    for col in df_numeric.columns:
        if df_numeric[col].std() > 0:
            valid_cols.append(col)
            # Додаємо fuzzy тільки для основних метрик (щоб не роздувати датасет)
            if 'bytes' in col or 'duration' in col or 'pkts' in col:
                c = df_numeric[col].mean()
                sigma = df_numeric[col].std()
                df_clean[f"{col}_fuzzy"] = fuzzify_data(df_numeric[col], c, sigma)

    # --- 5. Фіналізація ---
    # Повертаємо мітки
    df_clean["label"] = y

    return df_clean




