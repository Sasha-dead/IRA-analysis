# calculate_real_metrics.py
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings

warnings.filterwarnings('ignore')

import tensorflow as tf

tf.get_logger().setLevel('ERROR')

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import json
import sys

sys.path.append('.')
from hybrid_model import fuzzy_module, bayesian_module, train_lstm
from data_collector import preprocess


def calculate_real_metrics():
    """Основна функція для розрахунку метрик гібридної моделі."""
    print("=== РОЗРАХУНОК МЕТРИК ГІБРИДНОЇ МОДЕЛІ ===")

    # 1. ЗАВАНТАЖЕННЯ БІЛЬШЕ ДАНИХ
    try:
        df = pd.read_csv("conn4_log_labeled.csv", low_memory=False, nrows=20000)
        print(f"✓ Дані завантажено: {len(df)} рядків")
    except FileNotFoundError:
        print("✗ Файл conn4_log_labeled.csv не знайдено!")
        return None

    # 2. ПЕРЕДОБРОБКА
    try:
        processed = preprocess(df)
        print(f"✓ Дані оброблено: {len(processed)} рядків, {len(processed.columns)} ознак")
    except Exception as e:
        print(f"✗ Помилка при обробці даних: {e}")
        return None

    # 3. ЗНАЙТИ МІТКИ
    label_columns = ['anomaly', 'label', 'class', 'target', 'is_anomaly']
    found_label = next((col for col in label_columns if col in processed.columns), None)

    if not found_label:
        print("✗ Стовпець з мітками не знайдено!")
        return None

    y = processed[found_label].values
    X = processed.drop(found_label, axis=1).values

    # Статистика
    unique_labels, counts = np.unique(y, return_counts=True)
    label_dist = dict(zip(unique_labels, counts))
    print(f"✓ Розподіл міток: {label_dist}")
    print(f"✓ Дисбаланс: {max(counts) / min(counts):.1f}:1")

    # 4. НАВЧИТИ LSTM З ПОКРАЩЕНОЮ АРХІТЕКТУРОЮ
    print("📊 Навчання LSTM...")
    model, scaler = train_lstm(X, y, seq_length=10, epochs=50)

    # 5. ОТРИМАННЯ ПРОГНОЗІВ
    seq_length = 10
    n_samples = len(X) - seq_length + 1
    print(f"🔍 Генерація прогнозів для {n_samples} послідовностей...")

    # Швидкий батч-прогноз
    all_sequences = np.array([X[i:i + seq_length] for i in range(n_samples)])

    if scaler:
        all_sequences = scaler.transform(all_sequences.reshape(-1, X.shape[1])).reshape(n_samples, seq_length,
                                                                                        X.shape[1])

    # LSTM прогнози
    lstm_model = tf.keras.models.load_model('lstm_model.keras', compile=False)
    lstm_scores = lstm_model.predict(all_sequences, verbose=0, batch_size=64).flatten()

    # Fuzzy та Bayesian скори
    fuzzy_scores = []
    bayesian_score = bayesian_module({'Evidence': 1})

    for i in range(n_samples):
        seq = X[i:i + seq_length]
        features_mean = pd.Series(seq.mean(axis=0))
        fuzzy_scores.append(fuzzy_module(features_mean, [np.array([0.1, 0.5, 0.9])]))

    bayesian_scores = [bayesian_score] * n_samples

    # 6. ОБ'ЄДНАННЯ З РІЗНИМИ ВАГАМИ ТА ПОРОГОМ
    print("⚖️ Об'єднання результатів...")

    # ЕКСПЕРИМЕНТАЛЬНІ НАЛАШТУВАННЯ:
    w1, w2, w3 = 0.1, 0.1, 0.8
    threshold = 0.3

    y_pred_scores = []
    y_pred_binary = []

    for i in range(n_samples):
        combined = (fuzzy_scores[i] * w1 +
                    bayesian_scores[i] * w2 +
                    lstm_scores[i] * w3)

        y_pred_scores.append(combined)
        y_pred_binary.append(1 if combined > threshold else 0)

    # ПОСТ-ОБРОБКА: ФІЛЬТРАЦІЯ НИЗЬКОВПЕВНЕНИХ АНОМАЛІЙ
    print("🔧 Пост-обробка: фільтрація низьковпевнених аномалій...")

    confidence_threshold = 0.65  # ← ЕКСПЕРИМЕНТУЙТЕ З ЦИМ
    filtered = 0

    for i in range(len(y_pred_binary)):
        if y_pred_binary[i] == 1 and y_pred_scores[i] < confidence_threshold:
            y_pred_binary[i] = 0
            filtered += 1

    print(f"   Відфільтровано {filtered} низьковпевнених аномалій")

    # 7. ОБЧИСЛЕННЯ МЕТРИК
    y_true_trimmed = y[seq_length - 1:]

    accuracy = accuracy_score(y_true_trimmed, y_pred_binary)
    precision = precision_score(y_true_trimmed, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_trimmed, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_trimmed, y_pred_binary, zero_division=0)
    auc = roc_auc_score(y_true_trimmed, y_pred_scores)

    cm = confusion_matrix(y_true_trimmed, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()

    # 8. РЕЗУЛЬТАТИ
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТИ (покращена версія):")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.3f} ({accuracy * 100:.1f}%)")
    print(f"Precision: {precision:.3f} ({precision * 100:.1f}%)")
    print(f"Recall: {recall:.3f} ({recall * 100:.1f}%)")
    print(f"F1-Score: {f1:.3f} ({f1 * 100:.1f}%)")
    print(f"ROC-AUC: {auc:.3f}")
    print(f"\nМатриця помилок:")
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

    # Додаткова інформація
    total = tn + fp + fn + tp
    print(f"\nАналіз:")
    print(f"• Правильних прогнозів: {(tn + tp) / total * 100:.1f}%")
    print(f"• Пропущених аномалій (FN): {fn / (fn + tp) * 100:.1f}%")
    print(f"• Хибних тривог (FP): {fp / (tn + fp) * 100:.1f}%")

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(auc),
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    }


if __name__ == "__main__":
    calculate_real_metrics()