class Tensor:
    """Будет хранить выход из модели и иметь ссылку на саму модель

    array
        Массив данных, которые будут на выходе

    model (default=None)
        Модель, на которую будет ссылаться тензор
    """

    def __init__(self, array, model=None):
        self.array = array
        self.model = model

    def __repr__(self):
        return str(self.array)