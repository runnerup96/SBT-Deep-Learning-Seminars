import numpy as np
from nn.module.parameters import Parameters

class Linear:
    """Реализует линейный слой сети

    ---------
    Параметры
    ---------
    in_dim : int
        Размер входных данных

    out_dim : int
        Размер данных на выходе из слоя

    bias : bool (default=True)
        Использовать смещение или нет
    """
    def __init__(self, in_dim, out_dim, bias=True):
        self.in_dim = in_dim
        self.hid_dim = out_dim
        self.bias = bias

        self.W = Parameters((in_dim, out_dim))
        self.W._init_params()

        self.b = Parameters(out_dim)

        self.inpt = None

    def forward(self, inpt):
        """Реализует forward-pass

        ---------
        Параметры
        ---------
        inpt : np.ndarray, shape=(M, N_in)
            Входные данные

        ----------
        Возвращает
        ----------
        output : np.ndarray, shape=(M, N_out)
            Выход слоя
        """
        self.inpt = inpt
        # TODO: Реализовать forward pass для линейного слоя
        forward_pass = None

        return forward_pass

    def __call__(self, *inpt):
        """Аналогично forward"""
        return self.forward(*inpt)

    def parameters(self):
        """Возвращает параметры модели"""
        return (self.W, self.b)

    def _zero_grad(self):
        """Обнуляет градиенты модели"""
        self.W.grads = np.zeros(self.W.shape)
        self.b.grads = np.zeros(self.b.shape)

    def _compute_gradients(self, grads):
        """Считает градиенты модели"""
        # TODO: Реализовать рассчет градиентов для линейного слоя
        self.W.grads = None

        if self.bias:
            self.b.grads = None
        input_grads = None
        return input_grads

    def _train(self):
        """Переводит модель в режим обучения"""
        pass

    def _eval(self):
        """Переводит модель в режим оценивания"""
        pass

    def __repr__(self):
        return "Linear({}, {}, bias={})".format(self.in_dim, self.hid_dim,
                                                self.bias)