import numpy as np
from tensorflow.python.keras.datasets import mnist
np.random.seed(42)


def load_data(phase='train',
              validation_split=0.):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    assert phase in ['train', 'test']

    x = x_train if phase == 'train' else x_test
    y = y_train if phase == 'train' else y_test

    x = np.expand_dims(x, -1)

    if validation_split > 0. and phase == 'train':
        index = int(len(x)*(1.-validation_split))
        return (x[:index], y[:index]), (x[index:], y[index:])
    else:
        return x, y


def create_pair(x, y):
    formatted_x = []
    numbers_per_class = []
    nb_classes = len(np.unique(y))

    for l in range(nb_classes):
        formatted_x.append(x[y==l])
        numbers_per_class.append(len(x[y==l]))

    # same
    x1 = x[:len(x)//2]
    y1 = y[:len(x)//2]

    y2 = y1
    x2 = [formatted_x[y][np.random.randint(numbers_per_class[y])] for y in y2]

    # diff
    x1_ = x[len(x)//2:]
    y1_ = y[len(x)//2:]

    y2_ = (y1_ + np.random.randint(1, nb_classes, len(x)//2)) % nb_classes
    x2_ = [formatted_x[y][np.random.randint(numbers_per_class[y])] for y in y2_]

    x1 = np.concatenate((x1, x1_), axis=0)
    x2 = np.concatenate((x2, x2_), axis=0)
    y1 = np.concatenate((y1, y1_), axis=0)
    y2 = np.concatenate((y2, y2_), axis=0)

    return (x1, x2), (y1, y2)