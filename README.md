# Лабораторная работа №2 — Однослойный перцептрон

## Описание

Реализация однослойного перцептрона с нуля на Python (NumPy).  
Без использования PyTorch, TensorFlow, Keras, sklearn.linear_model.

---

## Структура проекта

    lab1/
    ├── model/
    │   └── perceptron.py     # класс перцептрона: sigmoid, forward, fit, predict, loss
    ├── data/
    │   └── dataset.py        # загрузка данных, стратификация, z-нормализация
    ├── extras/
    │   ├── generator.py      # генератор синтетических данных (доп. задание 1)
    │   ├── losses.py         # hinge loss и l2-регуляризация (доп. задание 2)
    │   └── metrics.py        # precision, recall, f1, roc-auc, roc-кривая (доп. задание 3)
    ├── utils/
    │   └── plots.py          # все функции для построения графиков
    ├── main.py               # точка входа, все эксперименты
    └── README.md

---
## Запуск

```bash
pip install numpy matplotlib scikit-learn
python main.py
```

---

## Обязательная часть

- Подготовка данных: `make_classification`, стратификация 70/30, z-нормализация
- Класс `Perceptron`: sigmoid, forward, compute_loss, fit (мини-батчи), predict
- Базовое обучение: η=0.1, epochs=100, batch_size=32
- Эксперименты: влияние lr, batch_size, инициализации весов

## Дополнительные задания

- **Задание 1** — собственный генератор данных (линейный, XOR, окружность, шум)
- **Задание 2** — Hinge loss и L2-регуляризация
- **Задание 3** — precision, recall, F1, ROC-AUC, ROC-кривая