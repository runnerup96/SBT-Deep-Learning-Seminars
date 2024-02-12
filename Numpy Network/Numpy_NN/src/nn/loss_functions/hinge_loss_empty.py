import numpy as np
from nn.loss_functions.loss import Loss


def hinge_loss(inpt, target):
    """Реализует функцию ошибки hinge loss

    ---------
    Параметры
    ---------
    inpt : Tensor
        Предсказание модели

    target
        Список реальных классов
        Одномерный массив

    ----------
    Возвращает
    ----------
    loss : Loss
        Ошибка
    """
    # Мы должны сконвертировать массив реальных меток
    # в двумерный массив размера (N, C),
    # где N -- число элементов
    # С -- число классов
    C = inpt.array.shape[-1]
    target = np.eye(C)[target]

    # TODO: Реализовать рассчет функции ошибки - loss
    # Можно взять такую реализацию - https://keras.io/api/losses/hinge_losses/#categoricalhinge-function
    loss = None

    # TODO: Реализовать рассчет градиента ошибки - grad
    grad = None

    return Loss(loss, grad, inpt.model)
