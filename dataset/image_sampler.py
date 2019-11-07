import tensorflow as tf
import numpy as np
from PIL import Image
import os


valid_exts = ['png', 'jpg', 'jpeg', 'bmp', 'PNG', 'JPG', 'JPEG', 'BMP']


class ImagePairSampler(object):
    def __init__(self, target_size=None,
                 normalize_mode='sigmoid',
                 is_training=True):
        self.target_size = target_size
        self.normalize_mode = normalize_mode
        self.is_training = is_training

    def flow(self, x, 
             y,
             batch_size=32,
             shuffle=True,
             seed=None):
        return Iterator(x,
            labels,
            self.target_size,
            batch_size,
            self.normalize_mode,
            shuffle,
            seed,
            self.is_training
        )

    def flow_from_directionary(self, image_dir,
                               batch_size=32,
                               shuffle=True,
                               seed=None,
                               nb_sample=None):
        class_dirs = sorted(os.listdir(image_dir))
        image_paths = []
        numbers_per_class = []
        for d in class_dirs:
            names = os.listdir(os.path.join(image_dir, d))
            numbers_per_class.append(len(names))
            for n in names:
                _, ext = os.path.splitext(n)
                ext = ext[1:]
                if ext in valid_exts:
                    image_paths.append(os.path.join(image_dir, d, n))

        labels = [[i] * numbers_per_class[i]
                  for i in range(len(class_dirs))]
        image_paths = np.array(image_paths)
        labels = [item for inner in labels for item in inner]
        labels = np.array(labels)

        return DirectoryIterator(
            image_paths,
            labels,
            self.target_size,
            batch_size,
            self.normalize_mode,
            shuffle,
            seed,
            self.is_training
        )

    def flow_from_tfrecord(self, file_paths,
                           batch_size=32,
                           shuffle=True,
                           seed=None):
        pass
    

class Iterator(tf.keras.preprocessing.image.Iterator):
    def __init__(self, x: np.array,
                 y: np.array,
                 target_size,
                 batch_size,
                 normalize_mode,
                 shuffle,
                 seed,
                 is_training):
        self.x = x
        self.y = y
        self.nb_classes = len(np.unique(y))
        self.target_size = target_size
        self.nb_sample = len(self.x)
        self.batch_size = batch_size
        self.normalize_mode = normalize_mode
        super().__init__(self.nb_sample, batch_size, shuffle, seed)
        self.is_training = is_training

        self.formatted_x = []
        self.numbers_per_class = []
        for l in range(self.nb_classes):
            self.formatted_x.append(self.x[self.y==l])
            self.numbers_per_class.append(len(self.x[self.y==l]))

    def __call__(self):
        if self.is_training:
            return self.flow_on_training()
        else:
            return self.flow_on_test()

    def flow_on_training(self):
        with self.lock:
            indices = next(self.index_generator)
        (x1, x2), (y1, y2) = self.sampling_same_class(indices[:self.batch_size//2])
        (x1_, x2_), (y1_, y2_) = self.sampling_different_class(indices[self.batch_size//2:])
        x1 = np.concatenate((x1, x1_), axis=0)
        x2 = np.concatenate((x2, x2_), axis=0)
        y1 = np.concatenate((y1, y1_), axis=0)
        y2 = np.concatenate((y2, y2_), axis=0)
        x1 = normalize(x1, self.normalize_mode)
        x2 = normalize(x2, self.normalize_mode)
        return (x1, x2), (y1, y2)
        
    def flow_on_test(self):
        raise NotImplementedError

    def sampling_same_class(self, indices):
        x1 = self.x[indices]
        y1 = self.y[indices]

        y2 = y1
        x2 = [self.formatted_x[y][np.random.randint(self.numbers_per_class[y])] for y in y2]

        return (x1, x2), (y1, y2)
        
    def sampling_different_class(self, indices):
        x1 = self.x[indices]
        y1 = self.y[indices]

        y2 = (y1 + np.random.randint(1, self.nb_classes, len(x1))) % self.nb_classes
        x2 = [self.formatted_x[y][np.random.randint(self.numbers_per_class[y])] for y in y2]
        
        return (x1, x2), (y1, y2)

class DirectoryIterator(Iterator):
    def __init__(self, paths,
                 labels,
                 target_size,
                 batch_size,
                 normalize_mode,
                 shuffle,
                 seed,
                 is_training):
        super().__init__(paths,
                         labels,
                         target_size,
                         batch_size,
                         normalize_mode,
                         shuffle,
                         seed,
                         is_training)
        self.current_paths = None

    def flow_on_training(self):
        with self.lock:
            indices = next(self.index_generator)
        (x1, x2), (y1, y2) = self.sampling_same_class(indices[:self.batch_size//2])
        (x1_, x2_), (y1_, y2_) = self.sampling_different_class(indices[self.batch_size//2:])
        x1 = np.concatenate((x1, x1_), axis=0)
        x2 = np.concatenate((x2, x2_), axis=0)
        y1 = np.concatenate((y1, y1_), axis=0)
        y2 = np.concatenate((y2, y2_), axis=0)

        x1 = np.array([Image.open(x) for x in x1])
        x2 = np.array([Image.open(x) for x in x2])

        x1 = normalize(x1, self.normalize_mode)
        x2 = normalize(x2, self.normalize_mode)
        return (x1, x2), (y1, y2)


def normalize(x, mode='tanh'):
    if mode == 'tanh':
        return (x.astype('float32') / 255 - 0.5) / 0.5
    elif mode == 'sigmoid':
        return x.astype('float32') / 255
    elif mode == None:
        return x
    else:
        raise NotImplementedError


ims = ImagePairSampler().flow_from_directionary('temp', batch_size=10)
print(ims())
