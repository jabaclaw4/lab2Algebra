# метрики качества и roc-кривая (доп. задание 3)

import numpy as np
import matplotlib.pyplot as plt


def confusion_matrix_values(y_true, y_pred):
    # считаем tp, tn, fp, fn вручную без sklearn
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp, tn, fp, fn


def accuracy(y_true, y_pred):
    # доля правильных ответов: (tp + tn) / (tp + tn + fp + fn)
    tp, tn, fp, fn = confusion_matrix_values(y_true, y_pred)
    return (tp + tn) / (tp + tn + fp + fn)


def precision(y_true, y_pred):
    # точность: tp / (tp + fp)
    tp, tn, fp, fn = confusion_matrix_values(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(y_true, y_pred):
    # полнота: tp / (tp + fn)
    tp, tn, fp, fn = confusion_matrix_values(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(y_true, y_pred):
    # f1: 2 * precision * recall / (precision + recall)
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def roc_curve(y_true, y_proba):
    # строим roc-кривую вручную — перебираем пороги
    thresholds = np.linspace(0, 1, 200)
    tprs = []
    fprs = []

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        tp, tn, fp, fn = confusion_matrix_values(y_true, y_pred)

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        tprs.append(tpr)
        fprs.append(fpr)

    return np.array(fprs), np.array(tprs)


def roc_auc(y_true, y_proba):
    # площадь под roc-кривой через метод трапеций
    fprs, tprs = roc_curve(y_true, y_proba)

    # сортируем по fpr для корректного подсчёта площади
    sorted_idx = np.argsort(fprs)
    fprs = fprs[sorted_idx]
    tprs = tprs[sorted_idx]

    return np.trapezoid(tprs, fprs)


def print_metrics(y_true, y_pred, y_proba=None):
    # выводим все метрики в консоль
    print(f"accuracy:  {accuracy(y_true, y_pred):.4f}")
    print(f"precision: {precision(y_true, y_pred):.4f}")
    print(f"recall:    {recall(y_true, y_pred):.4f}")
    print(f"f1-score:  {f1_score(y_true, y_pred):.4f}")

    if y_proba is not None:
        print(f"roc-auc:   {roc_auc(y_true, y_proba):.4f}")


def plot_roc_curve(y_true, y_proba, title='roc-кривая', filename=None):
    fprs, tprs = roc_curve(y_true, y_proba)
    auc = roc_auc(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fprs, tprs, color='#2980b9', linewidth=2, label=f'roc (auc = {auc:.3f})')

    # диагональ — случайный классификатор
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='случайный классификатор')

    ax.set_xlabel('fpr (false positive rate)')
    ax.set_ylabel('tpr (true positive rate)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=120)
    plt.show()


def plot_errors(model, X, y, title='ошибочно классифицированные точки', filename=None):
    # визуализируем неправильные предсказания на графике
    y_pred = model.predict(X)
    correct = y_pred == y
    wrong   = ~correct

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(X[correct, 0], X[correct, 1],
               c=['#2980b9' if c == 1 else '#e74c3c' for c in y[correct]],
               alpha=0.5, label='верно', edgecolors='none')

    ax.scatter(X[wrong, 0], X[wrong, 1],
               c='black', marker='x', s=80, linewidths=1.5, label='ошибка')

    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=120)
    plt.show()