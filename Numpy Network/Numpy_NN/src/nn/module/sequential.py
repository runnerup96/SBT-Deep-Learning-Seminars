from nn.module.tensor import Tensor

class Sequential:
    """Определяет структуру нейронной сети

    ---------
    Параметры
    ---------
    args
        Последовательность элементов нейронной сети
    """
    def __init__(self, *args):
        if len(args) == 0:
            msg = "В последовательности должен быть хотя бы один элемент"
            raise ValueError(msg)

        self.modules = args

    def forward(self, inpt):
        """Делает forward pass модели
        """
        for module in self.modules:
            inpt = module.forward(inpt)

        return Tensor(inpt, self)

    def __call__(self, *inpt):
        return self.forward(*inpt)

    def parameters(self):
        """Возвращает параметры всех модулей модели, генератор"""
        for module in self.modules:
            params = module.parameters()
            if isinstance(params, tuple):
                for param in params:
                    yield param
            else:
                yield params

    def zero_grad(self):
        """Обнуляет все накопленные градиенты"""
        for module in self.modules:
            module._zero_grad()

    def _compute_gradients(self, grads):
        """Считает градиенты всех элементов"""
        for module in reversed(self.modules):
            grads = module._compute_gradients(grads)

    def train(self):
        """Переводит модель в режим обучения"""
        for module in self.modules:
            module._train()

    def eval(self):
        """Переводит модель в режим оценки и предсказания"""
        for module in self.modules:
            module._eval()

    def __repr__(self):
        string = 'Sequential(\n\t'
        string += ',\n\t'.join(map(str, self.modules))
        string += '\n)'
        return string