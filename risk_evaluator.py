# risk_evaluator.py - Модуль оцінки та класифікації

import numpy as np
from scipy.special import softmax

def evaluate_risk(proc_score, fuzzy_adj=0.0, threshold=0.7):
    """Класифікація ризиків."""
    adjusted = proc_score + fuzzy_adj
    probs = softmax([adjusted, 1 - adjusted])  # Приклад бінарний
    class_idx = np.argmax(probs)
    classes = ['low', 'medium', 'high']
    risk_class = classes[min(class_idx + int(adjusted > 0.5), 2)]
    recommendations = {
        'low': 'Моніторинг не потрібен',
        'medium': 'Перевірити конфігурацію',
        'high': 'Негайно оновити firmware та шифрування'
    }
    return risk_class, recommendations[risk_class]