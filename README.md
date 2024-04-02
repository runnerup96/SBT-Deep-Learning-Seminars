# SBT-Deep-Learning-Seminars
Демонстрации/практикуму и ДЗ с семинарских занятий по глубокому обучению

### Установка окружения для работы
```
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
bash Anaconda3-2023.09-0-Linux-x86_64.sh
conda create -n dl_env python=3.10 pip

# Install your frameworks for suitable CUDA version
# get your torch version from here - https://pytorch.org/get-started/previous-versions/
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
# pytorch lightning from here https://pypi.org/project/pytorch-lightning/

# navigate to current dir and set up basic env libararies
cd SBT-Deep-Learning-Seminars
pip install -r requirements.txt
```

### Темы курса
* Пример проекта по глубокому обучению с лекции **Введение с DL** - https://github.com/runnerup96/minGPT
* Введение в PyTorch в глубоком обучении - `PyTorch tutorial/PyTorch intro.ipynb`
* Нейронная сеть на Numpy - `Numpy Network`
* Сверточные нейронные сети - `CNN`
* Задача сегментации - `Segmentation`
* Словарные эмбеддинги(word2vec, glove) - `word2vec`
* Машинный перевод - https://github.com/runnerup96/pytorch-machine-translation
