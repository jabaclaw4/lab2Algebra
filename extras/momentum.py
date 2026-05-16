#перцептрон с градиентным спуском с моментом (доп. задание 4)

import numpy as np


class PerceptronMomentum:
    def __init__(self, n_features, seed=42):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.01, n_features)
        self.b = 0.0
        self.train_losses = []
        self.val_losses = []

    def sigmoid(self, z):
        return np.where(
            z >= 0,
            1 / (1 + np.exp(-z)),
            np.exp(z) / (1 + np.exp(z))
        )
    def forward(self, X):
        return self.sigmoid(X @ self.w + self.b)

    def compute_loss(self, y_true, y_pred):
        # бинарная кросс-энтропия из теории препода
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def fit(self, X_train, y_train, X_val, y_val,
            epochs=100, lr=0.1, batch_size=32, beta=0.9):
        n = len(X_train)

        # инициализируем скорости (моменты) нулями
        v_w = np.zeros_like(self.w)
        v_b = 0.0

        for epoch in range(epochs):
            idx = np.random.permutation(n)
            X_sh = X_train[idx]
            y_sh = y_train[idx]

            for start in range(0, n, batch_size):
                Xb = X_sh[start:start + batch_size]
                yb = y_sh[start:start + batch_size]

                y_hat = self.forward(Xb)
                error = y_hat - yb
                # градиенты из теории препода
                grad_w = Xb.T @ error / len(yb)
                grad_b = error.mean()
                # обновление момента:
                v_w = beta * v_w + (1 - beta) * grad_w
                v_b = beta * v_b + (1 - beta) * grad_b
                self.w -= lr * v_w
                self.b -= lr * v_b

            self.train_losses.append(self.compute_loss(y_train, self.forward(X_train)))
            self.val_losses.append(self.compute_loss(y_val, self.forward(X_val)))

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)