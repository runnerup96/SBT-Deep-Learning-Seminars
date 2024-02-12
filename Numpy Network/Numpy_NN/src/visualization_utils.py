import numpy as np
import matplotlib.pyplot as plt

def visualize_cls_preds(X, model, th=0.5, borders=(-0.1, 0.5, 1.1), preds_dim=1,
                        colors=('tab:blue', 'tab:orange'), alpha=0.3):
    """Наглядная визуализация 2D двуклассовой классификации

    ---------
    Параметры
    ---------
    X
        Точки к которым деляются предсказания

    model
        Модель, с помощью которой делаются предсказания

    th : float (default=0.5)
        Трешхолд по которому делается отсечение

    borders : list or tuple (default=(-0.1, 0.5, 1.1))
        Границы по которым идет отсечение

    preds_dim : int (default=1)
        Размерность выхода предсказаний, если значение
        * 1, то будет класс 0, если значение меньше трешхолда,
          1 иначе
        * 2, то класс 0 будет соответствовать меньшему числу,
          а 1 большему

    colors : list or tuple (default=('tab:blue', 'tab:orange'))
        Цвета, которыми будет визуализироваться предсказание

    alpha : float (default=0.3)
        Степень прозрачности областей
    """
    plt.figure(figsize=(10, 7))

    preds = model(X).array
    if preds_dim == 1:
        preds = (preds.flatten() > th) * 1
    elif preds_dim == 2:
        preds = np.argmax(preds, axis=-1)

    for ind, cls in enumerate(sorted(set(preds))):
        plt.scatter(X[preds == cls][:, 0], X[preds == cls][:, 1], s=5,
                    label=str(cls), c=colors[ind])

    min_x, max_x = np.min(X[:, 0]), np.max(X[:, 0])
    min_y, max_y = np.min(X[:, 1]), np.max(X[:, 1])

    delta_x = (max_x - min_x) / 8
    delta_y = (max_y - min_y) / 8

    x = np.linspace(min_x - delta_x, max_x + delta_x, 100)
    y = np.linspace(min_y - delta_y, max_y + delta_y, 100)
    xx, yy = np.meshgrid(x, y)
    xy = np.vstack([xx.ravel(), yy.ravel()]).T
    preds_xy = model.forward(xy).array
    if preds_dim == 1:
        preds_xy = (preds_xy > th) * 1
    elif preds_dim == 2:
        preds_xy = np.argmax(preds_xy, axis=-1)
    preds_xy = preds_xy.reshape(xx.shape)
    plt.contourf(xx, yy, preds_xy, levels=borders,
                 colors=colors, alpha=0.3)

    plt.title('Что предсказалось')
    plt.legend()
    plt.show()


def plot_learning_curves(train_loss_history, valid_loss_history,
                         train_acc_history, valid_acc_history):
    """Строит графики качества на предсказании модели

    ---------
    Параметры
    ---------
    train_loss_history : list
        История ошибки на тренировочной выборке

    valid_loss_history : list
        История ошибки на валидационной выборке

    train_acc_history : list
        История точности на тренировочной выборке

    valid_acc_history : list
        История точности на валидационной выборке
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(list(range(1, 1 + len(train_loss_history))),
                 train_loss_history, c='tab:blue', label="Train")
    axes[0].plot(list(range(1, 1 + len(train_loss_history))),
                 valid_loss_history, c='tab:orange', label="Valid")

    axes[1].plot(list(range(1, 1 + len(train_acc_history))),
                 train_acc_history, c='tab:blue', label="Train")
    axes[1].plot(list(range(1, 1 + len(train_acc_history))),
                 valid_acc_history, c='tab:orange', label="Valid")

    axes[0].set_xlabel("Эпоха")
    axes[1].set_xlabel("Эпоха")

    axes[0].set_ylabel("Ошибка")
    axes[1].set_ylabel("Точность")

    axes[0].set_title("История ошибки")
    axes[1].set_title("История точности")

    axes[0].grid(ls=':')
    axes[1].grid(ls=':')

    axes[0].legend()
    axes[1].legend()

    plt.show()