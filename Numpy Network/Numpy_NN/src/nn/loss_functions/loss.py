class Loss:
    """Объект, возвращаемый функцией ошибки
    По сути контейнер, содержащий величину ошибки,
    ее градиент и ссылку на модель, чьи веса нужно обновить

    ---------
    Параметры
    ---------
    loss
        Ошибка

    grad
        Градиент

    model
        Ссылка на модель
    """
    def __init__(self, loss, grad, model):
        self.loss = loss
        self.grad = grad
        self.model = model

    def item(self):
        """Возвращает величину ошибки"""
        return self.loss

    def backward(self):
        """Запускает backpropagation"""
        self.model._compute_gradients(self.grad)

    def __repr__(self):
        return str(self.loss)