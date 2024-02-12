import numpy  as np

class GD:
    """Реализует обчный градиентный спуск

    ---------
    Параметры
    ---------
    params
        Параметры, передаваемые из модели

    lr : float (default=3e-4)
        Learning rate

    alpha1 : float (default=None)
        Если не None, то применяет l_1 регуляризацию
        с параметром alpha_1

    alpha2 : float (default=None)
        Если не None, то применяет l_2 регуляризацию
        с параметром alpha_2
    """
    def __init__(self, params, lr=3e-4, alpha1=None, alpha2=None):
        self.params = list(params)
        self.lr = lr
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def zero_grad(self):
        for param in self.params:
            param.grads = np.zeros(param.shape)

    def step(self):
        for param in self.params:
            grads = np.zeros(param.grads.shape)
            if not (self.alpha1 is None):
                grads += self.alpha1 * np.sign(param.params)
            if not (self.alpha2 is None):
                grads += self.alpha2 * param.params
            grads += param.grads

            param.params = param.params - self.lr * grads