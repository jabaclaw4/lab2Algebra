# функции для построения всех графиков
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt


def plot_loss_curves(models_dict, title='loss по эпохам', filename=None):
    # models_dict = {'название': model, ...}
    fig, ax = plt.subplots(figsize=(8, 4))

    for label, model in models_dict.items():
        ax.plot(model.train_losses, label=f'{label} train')
        ax.plot(model.val_losses, label=f'{label} val', linestyle='--')

    ax.set_xlabel('эпоха')
    ax.set_ylabel('loss')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=120)
    plt.show()


def plot_decision_boundary(model, X, y, title='разделяющая граница', filename=None):
    fig, ax = plt.subplots(figsize=(6, 5))

    # рисуем точки двух классов
    colors = ['#e74c3c', '#2980b9']
    for cls in [0, 1]:
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1],
                   c=colors[cls], label=f'класс {cls}',
                   alpha=0.6, edgecolors='k', linewidths=0.3)

    # строим разделяющую прямую w^T x + b = 0
    # из уравнения: w[0]*x1 + w[1]*x2 + b = 0
    # => x2 = -(w[0]*x1 + b) / w[1]
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x1_range = np.linspace(x1_min, x1_max, 200)

    if model.w[1] != 0:
        x2_range = -(model.w[0] * x1_range + model.b) / model.w[1]
        ax.plot(x1_range, x2_range, 'k-', linewidth=2, label='граница')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=120)
    plt.show()


def plot_accuracy_table(results, title='точность моделей'):
    # results = [{'label': '...', 'train_acc': 0.9, 'test_acc': 0.88}, ...]
    labels     = [r['label']     for r in results]
    train_accs = [r['train_acc'] for r in results]
    test_accs  = [r['test_acc']  for r in results]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, train_accs, width, label='train', color='#2980b9', alpha=0.8)
    ax.bar(x + width / 2, test_accs,  width, label='test',  color='#e74c3c', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel('accuracy')
    ax.set_title(title)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()