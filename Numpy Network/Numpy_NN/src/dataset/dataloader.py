import numpy as np

class Dataloader:
    """Загрузчик данных

    ---------
    Параметры
    ---------
    dataset : list
        Датасет, состоящий из пар (вектор, метка)

    batch_size : int (default=1000)
        Размер бача

    is_train : bool (default=True)
        Если True, перед выдачей перемешивает датасет
        Если False, то не перемешивает
    """
    def __init__(self, dataset, batch_size=1000, is_train=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.is_train = is_train
        self.array = list(range(len(self.dataset)))

        if is_train:
            np.random.shuffle(self.array)

    def __iter__(self):
        return self

    def __len__(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))

    def __next__(self):
        if len(self.array) == 0:
            raise StopIteration()

        if len(self.array) > self.batch_size:
            selected = self.array[:self.batch_size]
            self.array = self.array[self.batch_size:]
        else:
            selected = self.array[:]
            self.array = []

        data = []
        labels = []
        for ind in selected:
            data.append(self.dataset[ind][0])
            labels.append(self.dataset[ind][1])

        return np.array(data), labels