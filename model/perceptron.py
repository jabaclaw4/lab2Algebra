# класс однослойного перцептрона

import numpy as np


class Perceptron:
    def __init__(self, n_features, init='small_random', seed=42):
        rng = np.random.default_rng(seed)

        # три варианта инициализации для экспериментов
        if init == 'zeros':
            self.w = np.zeros(n_features)
        elif init == 'large_random':
            self.w = rng.normal(0, 10, n_features)
        else:
            # маленькие случайные веса по умолчанию
            self.w = rng.normal(0, 0.01, n_features)

        self.b = 0.0
        self.train_losses = []
        self.val_losses = []

    def sigmoid(self, z):
        # стабильный sigmoid, избегаем переполнения exp
        # формула из теории: σ(z) = 1 / (1 + e^(-z))
        return np.where(
            z >= 0,
            1 / (1 + np.exp(-z)),
            np.exp(z) / (1 + np.exp(z))
        )

    def forward(self, X):
        # z = w^T * x + b, затем sigmoid
        return self.sigmoid(X @ self.w + self.b)

    def compute_loss(self, y_true, y_pred):
        # бинарная кросс-энтропия из теории препода
        # L = -1/m * sum(y * log(ŷ) + (1 - y) * log(1 - ŷ))
        eps = 1e-12  # чтобы не было log(0)
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def fit(self, X_train, y_train, X_val, y_val, epochs=100, lr=0.1, batch_size=32):
        n = len(X_train)

        for epoch in range(epochs):
            # перемешиваем данные перед каждой эпохой
            idx = np.random.permutation(n)
            X_sh = X_train[idx]
            y_sh = y_train[idx]

            # проходим по мини-батчам
            for start in range(0, n, batch_size):
                Xb = X_sh[start:start + batch_size]
                yb = y_sh[start:start + batch_size]

                y_hat = self.forward(Xb)

                # градиенты из теории препода:
                # dL/dw = 1/m * X^T * (ŷ - y)
                # dL/db = 1/m * sum(ŷ - y)
                error = y_hat - yb
                grad_w = Xb.T @ error / len(yb)
                grad_b = error.mean()

                self.w -= lr * grad_w
                self.b -= lr * grad_b

            # сохраняем потери после каждой эпохи
            self.train_losses.append(self.compute_loss(y_train, self.forward(X_train)))
            self.val_losses.append(self.compute_loss(y_val, self.forward(X_val)))

    def predict(self, X):
        # порог 0.5 как указано в задании
        return (self.forward(X) >= 0.5).astype(int)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)