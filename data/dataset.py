import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def load_data():
    # генерируем датасет по параметрам из задания
    X, y = make_classification(
        n_samples=500,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=42,
        n_clusters_per_class=1
    )
    return X, y


def split_data(X, y, test_size=0.3, random_state=42):
    # стратификация сохраняет пропорцию классов в обоих выборках
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def normalize(X_train, X_test):
    # z-нормализация: считаем mean и std только по train
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0)
    X_train_norm = (X_train - mean) / std
    X_test_norm  = (X_test  - mean) / std

    return X_train_norm, X_test_norm