# точка входа — базовое обучение и все эксперименты

import numpy as np
from data.dataset import load_data, split_data, normalize
from model.perceptron import Perceptron
from utils.plots import plot_loss_curves, plot_decision_boundary, plot_accuracy_table
from extras.generator import make_gaussian, make_xor, make_circles
from extras.losses import PerceptronHinge, PerceptronL2
from extras.metrics import print_metrics, plot_roc_curve, plot_errors


# ─────────────────────────────────────────────
# подготовка данных
# ─────────────────────────────────────────────

X, y = load_data()
X_train, X_test, y_train, y_test = split_data(X, y)
X_train, X_test = normalize(X_train, X_test)


# ─────────────────────────────────────────────
# базовое обучение η=0.1, epochs=100, batch_size=32
# ─────────────────────────────────────────────

print("=== базовое обучение ===")
model_base = Perceptron(X_train.shape[1])
model_base.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=32)

print(f"train accuracy: {model_base.accuracy(X_train, y_train):.4f}")
print(f"test  accuracy: {model_base.accuracy(X_test,  y_test ):.4f}")

plot_loss_curves({'базовая': model_base}, title='базовое обучение')
plot_decision_boundary(model_base, X_test, y_test, title='разделяющая граница')


# ─────────────────────────────────────────────
# эксперимент 1 — влияние скорости обучения
# ─────────────────────────────────────────────

print("\n=== влияние скорости обучения ===")
lr_values = [0.001, 0.01, 0.5, 1.0]
lr_models  = {}
lr_results = []

for lr in lr_values:
    m = Perceptron(X_train.shape[1])
    m.fit(X_train, y_train, X_test, y_test, epochs=100, lr=lr, batch_size=32)
    lr_models[f'lr={lr}'] = m
    lr_results.append({
        'label':     f'lr={lr}',
        'train_acc': m.accuracy(X_train, y_train),
        'test_acc':  m.accuracy(X_test,  y_test),
    })
    print(f"lr={lr}: train={m.accuracy(X_train, y_train):.4f}  test={m.accuracy(X_test, y_test):.4f}")

plot_loss_curves(lr_models, title='влияние скорости обучения')
plot_accuracy_table(lr_results, title='точность при разных lr')


# ─────────────────────────────────────────────
# эксперимент 2 — влияние размера батча
# ─────────────────────────────────────────────

print("\n=== влияние размера батча ===")
batch_values = [1, 16, 64, 256]
batch_models  = {}
batch_results = []

for bs in batch_values:
    m = Perceptron(X_train.shape[1])
    m.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=bs)
    batch_models[f'bs={bs}'] = m
    batch_results.append({
        'label':     f'bs={bs}',
        'train_acc': m.accuracy(X_train, y_train),
        'test_acc':  m.accuracy(X_test,  y_test),
    })
    print(f"bs={bs}: train={m.accuracy(X_train, y_train):.4f}  test={m.accuracy(X_test, y_test):.4f}")

plot_loss_curves(batch_models, title='влияние размера батча')
plot_accuracy_table(batch_results, title='точность при разных batch_size')


# ─────────────────────────────────────────────
# эксперимент 3 — влияние инициализации весов
# ─────────────────────────────────────────────

print("\n=== влияние инициализации весов ===")
init_values = ['zeros', 'small_random', 'large_random']
init_models  = {}
init_results = []

for init in init_values:
    m = Perceptron(X_train.shape[1], init=init)
    m.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=32)
    init_models[init] = m
    init_results.append({
        'label':     init,
        'train_acc': m.accuracy(X_train, y_train),
        'test_acc':  m.accuracy(X_test,  y_test),
    })
    print(f"{init}: train={m.accuracy(X_train, y_train):.4f}  test={m.accuracy(X_test, y_test):.4f}")

plot_loss_curves(init_models, title='влияние инициализации весов')
plot_accuracy_table(init_results, title='точность при разных инициализациях')


# ─────────────────────────────────────────────
# доп. задание 1 — свой генератор данных
# ─────────────────────────────────────────────

print("\n=== доп. задание 1 — генератор данных ===")

datasets = {
    'линейный':   make_gaussian(noise=0.05),
    'xor':        make_xor(noise=0.05),
    'окружность': make_circles(noise=0.05),
}

for name, (Xg, yg) in datasets.items():
    Xg_tr, Xg_te, yg_tr, yg_te = split_data(Xg, yg)
    Xg_tr, Xg_te = normalize(Xg_tr, Xg_te)

    m = Perceptron(Xg_tr.shape[1])
    m.fit(Xg_tr, yg_tr, Xg_te, yg_te, epochs=100, lr=0.1, batch_size=32)

    print(f"{name}: train={m.accuracy(Xg_tr, yg_tr):.4f}  test={m.accuracy(Xg_te, yg_te):.4f}")
    plot_decision_boundary(m, Xg_te, yg_te, title=f'граница — {name}')


# ─────────────────────────────────────────────
# доп. задание 2 — hinge loss и l2-регуляризация
# ─────────────────────────────────────────────

print("\n=== доп. задание 2 — hinge loss ===")

# для hinge loss метки должны быть в {-1, +1}
y_train_hinge = np.where(y_train == 0, -1, 1)
y_test_hinge  = np.where(y_test  == 0, -1, 1)

model_hinge = PerceptronHinge(X_train.shape[1])
model_hinge.fit(X_train, y_train_hinge, X_test, y_test_hinge, epochs=100, lr=0.1, batch_size=32)
print(f"hinge — train={model_hinge.accuracy(X_train, y_train_hinge):.4f}  test={model_hinge.accuracy(X_test, y_test_hinge):.4f}")

print("\n=== доп. задание 2 — l2-регуляризация ===")
lam_values = [0.0, 0.01, 0.1, 1.0]
l2_models  = {}
l2_results = []

for lam in lam_values:
    m = PerceptronL2(X_train.shape[1], lam=lam)
    m.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=32)
    l2_models[f'λ={lam}'] = m
    l2_results.append({
        'label':     f'λ={lam}',
        'train_acc': m.accuracy(X_train, y_train),
        'test_acc':  m.accuracy(X_test,  y_test),
    })
    print(f"λ={lam}: train={m.accuracy(X_train, y_train):.4f}  test={m.accuracy(X_test, y_test):.4f}  ||w||={np.linalg.norm(m.w):.4f}")

plot_loss_curves(l2_models, title='влияние l2-регуляризации')
plot_accuracy_table(l2_results, title='точность при разных λ')


# ─────────────────────────────────────────────
# доп. задание 3 — метрики качества и roc
# ─────────────────────────────────────────────

print("\n=== доп. задание 3 — метрики качества ===")
y_pred  = model_base.predict(X_test)
y_proba = model_base.forward(X_test)

print_metrics(y_test, y_pred, y_proba)
plot_roc_curve(y_test, y_proba, title='roc-кривая базовой модели')
plot_errors(model_base, X_test, y_test, title='ошибочно классифицированные точки')