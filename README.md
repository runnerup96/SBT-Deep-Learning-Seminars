# SBT-Deep-Learning-Seminars

Репозиторий с материалами семинарских занятий по глубокому обучению, включающий демонстрации, практические работы и домашние задания.

## 📚 Описание курса

Этот курс охватывает фундаментальные концепции и практические аспекты глубокого обучения, начиная с базовых принципов и заканчивая современными архитектурами нейронных сетей.

## 🛠️ Установка и настройка окружения

### Предварительные требования
- Python 3.10+
- CUDA-совместимая видеокарта (опционально, для ускорения на GPU)
- Git

### Пошаговая установка

1. **Установка Anaconda**
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
bash Anaconda3-2023.09-0-Linux-x86_64.sh
```

2. **Создание виртуального окружения**
```bash
conda create -n dl_env python=3.10 pip
conda activate dl_env
```

3. **Установка PyTorch**
```bash
# Для CUDA 11.3
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

# Для CPU-only версии
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cpuonly -c pytorch
```

4. **Установка дополнительных зависимостей**
```bash
cd SBT-Deep-Learning-Seminars
pip install -r requirements.txt
```

## 📖 Структура курса

### 🎯 Основные модули

#### 1. **PyTorch Tutorial** (`PyTorch tutorial/`)
- Введение в PyTorch для глубокого обучения
- Основы работы с тензорами и автоматическим дифференцированием
- **Файл**: `PyTorch intro.ipynb`

#### 2. **Нейронные сети на NumPy** (`Numpy Network/`)
- Реализация нейронных сетей с нуля на NumPy
- Понимание принципов работы градиентного спуска
- Реализация различных слоев и функций активации
- **Файлы**: 
  - `Numpy Neural Network implementation.ipynb`
  - `numpy_neural_net_HW.ipynb`
  - `weight initialization retro.ipynb`

#### 3. **Сверточные нейронные сети** (`CNN/`)
- Архитектуры CNN (LeNet, AlexNet, VGG, ResNet)
- Реализация на PyTorch
- Аугментация данныx
- PyTorch Lightning для организации кода
- **Файлы**:
  - `seminar_cnn_1.ipynb` - CNN на PyTorch
  - `seminar_cnn_2.ipynb` - Арихитектуры CNN на PyTorch
  - `lightning.ipynb` - PyTorch Lightning
  - `albumentations.ipynb` - аугментация данных

#### 4. **Сегментация изображений** (`Segmentation/`)
- U-Net архитектура для сегментации
- Практическая реализация на PyTorch
- **Файл**: `unet_pytorch.ipynb`

#### 5. **Детекция объектов** (`Detection/`)
- Two-stage и One-stage модели для детекции объектов на изображениях
- Обучение модели из семейства R-CNN на PyTorch
- Практика обучения One-stage модели
- **Файл**: `object_detection.ipynb`

#### 6. **Рекуррентные нейронные сети** (`RNN implementation/`)
- Реализация LSTM с нуля
- Стратегии сэмплирования для генерации текста
- **Файлы**:
  - `LSTM.ipynb` - реализация LSTM
  - `Sampling strategies.ipynb` - стратегии сэмплирования

#### 7. **Словарные эмбеддинги** (`word2vec/`)
- Реализация Word2Vec с нуля
- Визуализация эмбеддингов с помощью Gensim
- **Файлы**:
  - `word2vec_implementation.ipynb` - реализация Word2Vec
  - `Gensim word vector visualization.ipynb` - визуализация

#### 8. **Трансформеры** (`Transformers/`)
- Механизм внимания (Attention)
- BPE токенизация
- Онлайн softmax
- **Файлы**:
  - `self_attention_demo.ipynb` - демонстрация self-attention
  - `BPE_tokenization_demo.ipynb` - BPE токенизация
  - `Online softmax.ipynb` - онлайн softmax

#### 9. **Большие языковые модели** (`LLMs/`)
- Тюнинг BERT и T5
- Законы масштабирования (Scaling Laws)
- **Файлы**:
  - `bert_tuning.ipynb` - тюнинг BERT
  - `t5_tuning.ipynb` - тюнинг T5
  - `Scaling laws.ipynb` - законы масштабирования

### 🏠 Домашние задания

#### **HW1: Нейронная сеть на NumPy** (`HW1-Numpy-Network/`)
- Полная реализация нейронной сети с нуля
- Включает все необходимые компоненты: слои, функции активации, оптимизаторы
- **Структура**:
  - `src/` - исходный код
  - `tests/` - тесты для проверки корректности
  - `run.py` - скрипт запуска
  - `solution.py` - решение

#### **HW2** (`HW2/`)
- Практические задания по глубокому обучению
- **Файлы**:
  - `HW2_baseline.ipynb` - базовое решение
  - `HW2_sbt.ipynb` - основное задание

## 🚀 Быстрый старт

1. **Клонируйте репозиторий**
```bash
git clone <repository-url>
cd SBT-Deep-Learning-Seminars
```

2. **Настройте окружение** (см. раздел "Установка и настройка окружения")

3. **Запустите Jupyter Notebook**
```bash
jupyter notebook
```

4. **Начните с вводного материала**
   - Откройте `PyTorch tutorial/PyTorch intro.ipynb`
   - Изучите основы PyTorch

## 📋 Рекомендуемый порядок изучения

1. **PyTorch Tutorial** - основы фреймворка
2. **Numpy Network** - понимание принципов работы нейронных сетей
3. **CNN** - сверточные нейронные сети
4. **RNN implementation** - рекуррентные сети
5. **word2vec** - эмбеддинги
6. **Transformers** - современные архитектуры
7. **LLMs** - большие языковые модели
8. **Segmentation** - специализированные задачи

## 🔗 Дополнительные ресурсы

- **Пример проекта по глубокому обучению**: [minGPT](https://github.com/runnerup96/minGPT)
- **Машинный перевод**: [pytorch-machine-translation](https://github.com/runnerup96/pytorch-machine-translation)

## 📝 Лицензия

Материалы предназначены для образовательных целей.

## 🤝 Вклад в проект

Если вы нашли ошибки или хотите улучшить материалы, создайте issue или pull request.

---

**Примечание**: Убедитесь, что у вас установлена совместимая версия CUDA для работы с GPU. Для получения актуальных версий PyTorch посетите [официальный сайт](https://pytorch.org/get-started/previous-versions/).
