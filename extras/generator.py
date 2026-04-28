# генератор синтетических данных для бинарной классификации (доп. задание 1)

import numpy as np


def make_gaussian(n_samples=500, centers=None, cov=None, noise=0.0, seed=42):
    # два гауссовых облака с заданными центрами — линейно разделимые данные
    rng = np.random.default_rng(seed)

    if centers is None:
        centers = [[-2, -2], [2, 2]]
    if cov is None:
        cov = [[1, 0], [0, 1]]

    n_each = n_samples // 2

    X0 = rng.multivariate_normal(centers[0], cov, n_each)
    X1 = rng.multivariate_normal(centers[1], cov, n_each)

    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_each), np.ones(n_each)])

    # добавляем шум — сдвигаем метку с вероятностью noise
    if noise > 0:
        flip_mask = rng.random(len(y)) < noise
        y[flip_mask] = 1 - y[flip_mask]

    return X, y.astype(int)


def make_xor(n_samples=500, noise=0.0, seed=42):
    # нелинейно разделимые данные — точки в углах квадрата (xor)
    # перцептрон не может разделить такие данные прямой линией
    rng = np.random.default_rng(seed)

    n_each = n_samples // 4

    # четыре угла: (-, -), (+, +) -> класс 0 | (-, +), (+, -) -> класс 1
    X = np.vstack([
        rng.normal([-1, -1], 0.3, (n_each, 2)),
        rng.normal([ 1,  1], 0.3, (n_each, 2)),
        rng.normal([-1,  1], 0.3, (n_each, 2)),
        rng.normal([ 1, -1], 0.3, (n_each, 2)),
    ])
    y = np.hstack([
        np.zeros(n_each * 2),
        np.ones(n_each * 2)
    ])

    if noise > 0:
        flip_mask = rng.random(len(y)) < noise
        y[flip_mask] = 1 - y[flip_mask]

    return X, y.astype(int)


def make_circles(n_samples=500, radius=1.5, noise=0.0, seed=42):
    # нелинейно разделимые данные — точки внутри и снаружи окружности
    rng = np.random.default_rng(seed)

    angles = rng.uniform(0, 2 * np.pi, n_samples)
    radii  = rng.uniform(0, radius * 1.8, n_samples)

    X = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    # класс 1 — внутри окружности, класс 0 — снаружи
    y = (radii <= radius).astype(int)

    if noise > 0:
        flip_mask = rng.random(len(y)) < noise
        y[flip_mask] = 1 - y[flip_mask]

    return X, y.astype(int)