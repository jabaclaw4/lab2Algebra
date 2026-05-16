# 5-кратная кросс-валидация для подбора гиперпараметров

import numpy as np
from model.perceptron import Perceptron


def k_fold_split(X, y, k=5, seed=42):
    # делим индексы на k равных частей
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    folds = np.array_split(idx, k)
    return folds


def cross_validate(X, y, lr, batch_size, epochs=100, k=5):
    # обучаем модель k раз, каждый раз на разных fold-ах
    folds = k_fold_split(X, y, k=k)
    fold_accuracies = []

    for i in range(k):
        val_idx   = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # нормализация внутри каждого fold отдельно
        mean = X_tr.mean(axis=0)
        std  = X_tr.std(axis=0)
        X_tr  = (X_tr  - mean) / std
        X_val = (X_val - mean) / std

        model = Perceptron(X_tr.shape[1])
        model.fit(X_tr, y_tr, X_val, y_val,
                  epochs=epochs, lr=lr, batch_size=batch_size)

        fold_accuracies.append(model.accuracy(X_val, y_val))

    mean_acc = np.mean(fold_accuracies)
    std_acc  = np.std(fold_accuracies)
    return mean_acc, std_acc


def grid_search(X, y, lr_values, batch_values, epochs=100, k=5):
    # перебираем все комбинации lr и batch_size
    results = []

    for lr in lr_values:
        for bs in batch_values:
            mean_acc, std_acc = cross_validate(X, y, lr=lr, batch_size=bs, epochs=epochs, k=k)
            results.append({
                'lr':       lr,
                'batch':    bs,
                'mean_acc': mean_acc,
                'std_acc':  std_acc,
            })
            print(f"lr={lr}  bs={bs}: mean={mean_acc:.4f}  std={std_acc:.4f}")

    # находим лучшую комбинацию по средней точности
    best = max(results, key=lambda r: r['mean_acc'])
    return results, best


def train_final_model(X, y, lr, batch_size, epochs=100):
    # обучаем финальную модель на всех данных с лучшими параметрами
    mean = X.mean(axis=0)
    std  = X.std(axis=0)
    X_norm = (X - mean) / std

    model = Perceptron(X_norm.shape[1])
    model.fit(X_norm, y, X_norm, y, epochs=epochs, lr=lr, batch_size=batch_size)
    return model, mean, std