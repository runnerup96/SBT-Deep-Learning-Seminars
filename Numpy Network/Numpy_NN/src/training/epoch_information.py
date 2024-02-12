class EpochInformation:
    """Выдает информацию об эпохе

    ---------
    Параметры
    ---------
    max_num : int
        Максимальное число эпох

    description : list
        Список, каждый элемент которого представляет собой кортеж:
        * первый элемент содержит название параметра
        * второй элемент определяет максимальное число символов в строке
          если это число меньше длины названия, берется длина названия
          если в строчку не помещается описание величины,
          все доступные символы заменятся на символы '?'
        * третий элемент (необязателен) определяет число знаков
          после запятой для вещественных чисел
    """

    def __init__(self, max_num, description):
        self.max_num = max_num
        self.description = description
        self.iteration = 0

        for ind in range(len(description)):
            info = description[ind]
            if len(info[0]) > info[1]:
                if len(info) == 2:
                    self.description[ind] = (info[0], len(info[0]))
                else:
                    self.description[ind] = (info[0], len(info[0]), info[2])

        print('|{0:^{1}}|'.format('Epoch',
                                  max(5, 1 + 2 * len(str(max_num)))),
              end='')

        for info in self.description:
            print('{0:^{1}}|'.format(info[0], info[1]), end='')

        print()

        print('|{0:=>{1}}|'.format('', max(5, 1 + 2 * len(str(max_num)))),
              end='')
        for info in self.description:
            print('{0:=>{1}}|'.format('', info[1]), end='')

        print()

    def update(self, values, step=1):
        """Выводит параметры для следующей эпохи

        ---------
        Параметры
        ---------
        values : dict
            Словарь, в котором ключи - это названия из description,
            а значения - это то, что нужно вписать в ячейку

        step : int (default=1)
            Количество сделанных итераций с последнего вызова
        """
        self.iteration += step

        print('|{0:>{1}}/{2}|'.format(self.iteration,
                                      max(4 - len(str(self.max_num)),
                                          len(str(self.max_num))),
                                      self.max_num), end='')

        for info in self.description:
            if len(info) == 2:
                string = str(values[info[0]])
            else:
                string = '{0:.{1}f}'.format(values[info[0]], info[2])

            if len(string) > info[1]:
                string = '?' * info[1]

            print('{0:^{1}}|'.format(string, info[1]), end='')

        print()