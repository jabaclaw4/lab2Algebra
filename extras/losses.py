# альтернативные функции потерь и регуляризация

import numpy as np


class PerceptronHinge:
    #перцептрон с hinge loss, метки y должны быть в {-1, +1}
    def __init__(self, n_features, seed=42):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.01, n_features)
        self.b = 0.0
        self.train_losses = []
        self.val_losses = []

    def forward(self, X):
        return X @ self.w + self.b

    def hinge_loss(self, y_true, y_raw):
        # y_true должен быть в {-1, +1}
        return np.mean(np.maximum(0, 1 - y_true * y_raw))

    def fit(self, X_train, y_train, X_val, y_val, epochs=100, lr=0.1, batch_size=32):
        n = len(X_train)
        for epoch in range(epochs):
            idx = np.random.permutation(n)
            X_sh = X_train[idx]
            y_sh = y_train[idx]

            for start in range(0, n, batch_size):
                Xb = X_sh[start:start + batch_size]
                yb = y_sh[start:start + batch_size]

                y_raw = self.forward(Xb)
                mask = (yb * y_raw) < 1
                grad_w = -np.mean((yb[mask, None] * Xb[mask]), axis=0) if mask.any() else np.zeros_like(self.w)
                grad_b = -np.mean(yb[mask]) if mask.any() else 0.0

                self.w -= lr * grad_w
                self.b -= lr * grad_b

            self.train_losses.append(self.hinge_loss(y_train, self.forward(X_train)))
            self.val_losses.append(self.hinge_loss(y_val, self.forward(X_val)))

    def predict(self, X):
        # предсказание по знаку выхода
        return np.sign(self.forward(X)).astype(int)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)


class PerceptronL2:
    # перцептрон с кросс-энтропией и l2-регуляризацией
    def __init__(self, n_features, lam=0.01, seed=42):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.01, n_features)
        self.b = 0.0
        self.lam = lam#коэффициент регуляризации λ
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
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)
        bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        l2  = (self.lam / 2) * np.sum(self.w ** 2)
        return bce + l2

    def fit(self, X_train, y_train, X_val, y_val, epochs=100, lr=0.1, batch_size=32):
        n = len(X_train)

        for epoch in range(epochs):
            idx = np.random.permutation(n)
            X_sh = X_train[idx]
            y_sh = y_train[idx]
            for start in range(0, n, batch_size):
                Xb = X_sh[start:start + batch_size]
                yb = y_sh[start:start + batch_size]
                y_hat = self.forward(Xb)
                error = y_hat - yb

                grad_w = Xb.T @ error / len(yb) + self.lam * self.w
                grad_b = error.mean()

                self.w -= lr * grad_w
                self.b -= lr * grad_b

            self.train_losses.append(self.compute_loss(y_train, self.forward(X_train)))
            self.val_losses.append(self.compute_loss(y_val, self.forward(X_val)))

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)