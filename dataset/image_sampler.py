import tensorflow as tf
import numpy as np
from PIL import Image
import os


class ImagePairSampler(object):
    def __init__(self, target_size=None,
                 normalize_mode='tanh',
                 is_training=True):
        self.target_size = target_size
        self.normalize_mode = normalize_mode
        self.is_training = is_training

    def flow(self, x, 
             y=None,
             batch_size=32,
             shuffle=True,
             seed=None):
        pass

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
    
            
class DirectoryIterator(tf.keras.preprocessing.image.Iterator):
    def __init__(self, paths,
                 labels,
                 target_size,
                 batch_size,
                 normalize_mode,
                 shuffle,
                 seed,
                 is_training):
        self.paths = paths
        self.labels = labels
        self.nb_classes = len(np.unique(labels))
        self.target_size = target_size
        self.nb_sample = len(self.paths)
        self.batch_size = batch_size
        self.normalize_mode = normalize_mode
        super().__init__(self.nb_sample, batch_size, shuffle, seed)
        self.current_paths = None
        self.is_training = is_training

        self.formatted_paths = []
        self.numbers_per_class = []
        for l in range(self.nb_classes):
            self.formatted_paths.append(self.paths[self.labels==l])
            self.numbers_per_class.append(len(self.paths[self.labels==l]))

    def random_sampling(self, same_class=True):
        if same_class:
            return self.sampling_same_class()
        else:
            return self.sampling_differnt_class()

    def sampling_same_class(self):
        y1 = np.random.randint(0, self.nb_classes, self.batch_size)
        x1 = [self.formatted_paths[y][np.random.randint(self.numbers_per_class[y])] for y in y1]

        y2 = y1
        x2 = [self.formatted_paths[y][np.random.randint(self.numbers_per_class[y])] for y in y2]

        return (x1, x2), (y1, y2)
        
    def sampling_differnt_class(self):
        y1 = np.random.randint(0, self.nb_classes, self.batch_size)
        x1 = [self.formatted_paths[y][np.random.randint(self.numbers_per_class[y])] for y in y1]

        y2 = (y1 + np.random.randint(1, self.nb_classes, self.batch_size)) % self.nb_classes
        x2 = [self.formatted_paths[y][np.random.randint(self.numbers_per_class[y])] for y in y2]
        
        return (x1, x2), (y1, y2)

    
class TFRecordIterator:
    def __init__(self, file_paths,
                 map_fn=None,
                 compression_type='GZIP',
                 target_size=None,
                 color_mode='rgb',
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 is_training=True):
        self.file_paths = file_paths
        self.nb_sample = 0
        if map_fn is not None:
            self.map_dataset = map_fn
        self.compression_type = compression_type
        self._target_size = target_size
        self.color_mode = color_mode
        self._batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.is_training = is_training
        self._build_dataset()

    def __next__(self):
        return next(self.iterator)

    def __call__(self):
        return next(self.iterator)

    def __iter__(self):
        return self.iterator

    @staticmethod
    def data_to_image(x: tf.Tensor):
        if x.get_shape().as_list()[-1] == 1:
            x = x.reshape(x.get_shape().as_list()[:-1])
        x = np.array(x)
        return denormalize(x)

    def map_dataset(self, serialized):
        features = {
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'channel': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string),
        }
        features = tf.parse_single_example(serialized, features)
        height = tf.cast(features['height'], tf.int32)
        width = tf.cast(features['width'], tf.int32)
        channel = tf.cast(features['channel'], tf.int32)
        images = tf.decode_raw(features['image'], tf.uint8)
        images = tf.cast(images, tf.float32)
        images = tf.reshape(images, (height, width, channel))
        images = tf.image.resize_images(images, self._target_size)
        return (images / 255 - 0.5) / 0.5

    def _build_dataset(self):
        self.dataset = tf.data.TFRecordDataset(self.file_paths, compression_type=self.compression_type)
        self.dataset = self.dataset.repeat(1 if not self.is_training else -1)
        self.dataset = self.dataset.map(self.map_dataset)
        if self.shuffle:
            self.dataset = self.dataset.shuffle(buffer_size=256, seed=self.seed)
        self.dataset = self.dataset.batch(self._batch_size)
        self.iterator = self.dataset.make_one_shot_iterator()

    def __len__(self):
        if self.nb_sample:
            return self.nb_sample
        else:
            print('\nComputing the number of records now.'),
            print('It takes a long time only for the 1st time\n')
            options = tf.python_io.TFRecordOptions(
                tf.python_io.TFRecordCompressionType.GZIP) if self.compression_type == 'GZIP' \
                else None
            for path in self.file_paths:
                self.nb_sample += sum(1 for _ in tf.python_io.tf_record_iterator(path, options=options))
        return self.nb_sample

    @property
    def target_size(self):
        return self._target_size

    @target_size.setter
    def target_size(self, target_size):
        self._target_size = target_size
        self._build_dataset()

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        self._batch_size = batch_size
        self._build_dataset()


ims = ImagePairSampler().flow_from_directionary('temp', batch_size=10)
ims.random_sampling(False)
