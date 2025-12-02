# hybrid_model.py - Модуль гібридної моделі
import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from deap import base, creator, tools, algorithms
import skfuzzy as fuzz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score
import os

# --- ДОДАТИ ЦЕ ВІДРАЗУ ПІСЛЯ ІМПОРТІВ (ГЛОБАЛЬНО) ---
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)


# ----------------------------------------------------

def fuzzy_module(features, rules):
    """Fuzzy-модуль: обчислення fuzzy_score."""
    universe = np.arange(0, 1, 0.01)
    low = fuzz.trimf(universe, [0, 0, 0.3])
    medium = fuzz.trimf(universe, [0.2, 0.5, 0.8])
    high = fuzz.trimf(universe, [0.7, 1, 1])

    scores = []
    for rule in rules:
        interp_low = fuzz.interp_membership(universe, low, features.mean())
        interp_medium = fuzz.interp_membership(universe, medium, features.mean())
        interp_high = fuzz.interp_membership(universe, high, features.mean())
        scores.append(np.max([interp_low, interp_medium, interp_high]) * rule.mean())
    return np.mean(scores)


def bayesian_module(evidence):
    """Bayesian-модуль: обчислення P_adj."""
    model = DiscreteBayesianNetwork([('Threat', 'Evidence')])
    cpd_threat = TabularCPD('Threat', 2, [[0.7], [0.3]])
    cpd_evidence = TabularCPD('Evidence', 2, [[0.9, 0.2], [0.1, 0.8]],
                              evidence=['Threat'], evidence_card=[2])
    model.add_cpds(cpd_threat, cpd_evidence)
    model.check_model()
    infer = VariableElimination(model)
    P = infer.query(variables=['Threat'], evidence=evidence).values[1]
    return P * 0.5


def train_lstm(X, y, seq_length=10, epochs=50):
    """Навчання LSTM на реальних даних."""
    # Нормалізація
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Створення послідовностей (ковзне вікно)
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - seq_length):
        X_seq.append(X_scaled[i:i + seq_length])
        y_seq.append(y[i + seq_length])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # Розділення
    X_train, X_val, y_train, y_val = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42
    )

    # Ваги класів для дисбалансу
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    # Модель
    model = Sequential()
    model.add(Bidirectional(LSTM(100, return_sequences=True),
                            input_shape=(seq_length, X_seq.shape[2])))
    model.add(LSTM(50))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Рання зупинка
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=epochs, batch_size=32,
              validation_data=(X_val, y_val),
              class_weight=class_weight_dict, callbacks=[early_stop], verbose=1)

    model.save('lstm_model.keras')
    print("✅ LSTM модель навчена та збережена як 'lstm_model.keras'")
    return model, scaler


def lstm_module(seq, scaler=None):
    """LSTM-модуль: прогноз аномалій."""
    if os.path.exists('lstm_model.keras'):
        model = load_model('lstm_model.keras', compile=False)
    else:
        if os.path.exists('lstm_model.h5'):
            model = load_model('lstm_model.h5', compile=False)
            print("⚠ Використовується застарілий формат .h5")
        else:
            raise FileNotFoundError("LSTM модель не знайдено. Спочатку навчіть модель за допомогою train_lstm.")

    if scaler:
        seq = scaler.transform(seq.reshape(-1, seq.shape[-1])).reshape(seq.shape)

    pred = model.predict(seq, verbose=0)
    return pred.mean()


def optimize_weights(fitness_func, dim=3):
    """Оптимізація ваг PSO з DEAP."""
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=dim)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", fitness_func)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    pop = toolbox.population(n=50)
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, verbose=False)
    best = tools.selBest(pop, 1)[0]
    return best


def hybrid_score(features, seq, evidence, rules, scaler=None):
    """Інтеграція: proc_score."""
    fuzzy_score = fuzzy_module(features, rules)
    P_adj = bayesian_module(evidence)
    pred = lstm_module(seq, scaler)

    def fitness(w):
        combined = fuzzy_score * w[0] + P_adj * w[1] + pred * w[2]
        return (abs(combined - 0.5),)

    w = optimize_weights(fitness)
    return sum(w[i] * s for i, s in enumerate([fuzzy_score, P_adj, pred]))