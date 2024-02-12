from nn.loss_functions.loss import Loss

def mse_loss(inpt, target):
    """Реализует функцию ошибки

    ---------
    Параметры
    ---------
    inpt : Tensor
        Предсказание модели

    target
        Целевая функция

    ----------
    Возвращает
    ----------
    loss : Loss
        Ошибка
    """
    loss = (inpt.array - target) ** 2
    loss = loss / inpt.array.shape[-1]
    loss = loss.sum(-1)
    loss = loss / inpt.array.shape[0]
    loss = loss.sum()

    grad = 2 * (inpt.array - target)
    grad = grad / inpt.array.shape[-1]
    grad = grad / inpt.array.shape[0]

    return Loss(loss, grad, inpt.model)