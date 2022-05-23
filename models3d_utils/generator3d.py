import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os.path import join
from os import listdir
import numpy as np
import random
from tensorflow.keras.utils import Sequence


def prepare_images3d_batch(images_batch):
    images_batch = images_batch / 255.0
    images_batch = images_batch[None, :, :, :]
    images_batch = tf.transpose(
        images_batch, perm=[0, 2, 3, 1, 4],
    )
    return images_batch


def prepare_masks3d_batch(masks_batch):
    masks_batch = masks_batch / 255.0
    masks_batch = tf.where(condition=tf.math.greater(masks_batch, 0.5), x=1.0, y=0.0)
    masks_batch = masks_batch[None, :, :, :]
    masks_batch = tf.transpose(
        masks_batch, perm=[0, 2, 3, 1, 4],
    )
    return masks_batch


class ImageMaskGenerator3d:
    def __init__(
            self, main_dir_path, imgs_dir_name, masks_dir_name, subdirs_names,
            color_mode="grayscale", target_size=(512, 512),
            interpolation="bilinear", aug_dict={},
            batch_size=4, shuffle=True, seed=7
    ):
        images_datagen = ImageDataGenerator(**aug_dict)
        masks_datagen = ImageDataGenerator(**aug_dict)

        self.images_generator = images_datagen.flow_from_directory(
            join(main_dir_path, imgs_dir_name),
            target_size=target_size,
            color_mode=color_mode,
            classes=subdirs_names,
            class_mode=None,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            interpolation=interpolation,
            # ToDo - label_mode=None, #Float32
        )
        self.masks_generator = masks_datagen.flow_from_directory(
            join(main_dir_path, masks_dir_name),
            target_size=target_size,
            color_mode=color_mode,
            classes=subdirs_names,
            class_mode=None,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            interpolation=interpolation,
            # ToDo - label_mode=None, #Float32
        )

    def __iter__(self):
        images_masks_generator = zip(self.images_generator, self.masks_generator)

        for (images_batch, masks_batch) in images_masks_generator:
            images_batch = prepare_images3d_batch(images_batch)
            masks_batch = prepare_masks3d_batch(masks_batch)

            idx_image = (self.images_generator.batch_index - 1) * self.images_generator.batch_size

            self.files_names_batch = self.images_generator.filenames[
                                     idx_image: idx_image + self.images_generator.batch_size
                                     ]

            yield (images_batch, masks_batch)


def get_class_name(file_path):
    i = 0
    while (file_path[i]).isdigit():
        i += 1
    return file_path[:i]


# ToDo - channels(by color_mode)
# ToDo - interpolation
# ToDo - batch_size (Сейчас batch_size это количество снимков в одном батче)
class ImageMaskFlow3d(Sequence):
    def __init__(
            self, main_dir_path, imgs_dir_name, masks_dir_name,
            color_mode="grayscale", target_size=(512, 512),
            interpolation="bilinear",
            batch_size=4, shuffle=True, seed=7,
    ):
        self.imgs_patch = join(main_dir_path, imgs_dir_name)
        self.masks_patch = join(main_dir_path, masks_dir_name)
        self.channels = 1 if color_mode == "grayscale" else 3
        self.img_size = target_size
        self.interpolation = interpolation
        self.batch_size = batch_size
        # ToDo:
        '''
            class должен определять не 2D область, а 3D область максимального размера
            Т.е. использовать алгоритм сканирующего окна также по измерению глубины
        '''
        self.classes = sorted(listdir(self.imgs_patch), key=lambda name: int(name))
        # print(self.classes)
        self.num_classes = len(self.classes)
        self.imgs_by_classes = []

        for i_class in range(self.num_classes):
            self.imgs_by_classes.append(
                np.array([
                    join(self.classes[i_class], file_name)
                    for file_name in sorted(
                        listdir(join(self.imgs_patch, self.classes[i_class])),
                        key=lambda file_name: int(file_name[:file_name.find('.')])
                    )]))

        self.imgs_by_classes = np.array(self.imgs_by_classes)
        if shuffle:
            self.imgs_by_classes = self.imgs_by_classes.reshape(-1, self.batch_size)
            random.seed(seed)  # Set random seed for Python
            np.random.seed(seed)  # Set random seed for numpy
            tf.random.set_seed(seed)  # tf cpu fix seed
            self.batch_idxs = np.random.permutation(range(self.imgs_by_classes.shape[0]))
        else:
            # Если перемешивания нет - считываю последовательно,
            self.imgs_by_classes = self.imgs_by_classes.reshape(self.num_classes, -1, self.batch_size)
            self.imgs_by_classes = self.imgs_by_classes.transpose((1, 0, 2)).reshape(-1, self.batch_size)
            # Считывать последовательно классы
            self.batch_idxs = list(range(self.imgs_by_classes.shape[0]))

        # print(self.batch_idxs)
        self.num_batches = self.imgs_by_classes.shape[0]

        # Проверим, что все снимки в батче относятся к одному классу (ToDo - объёму)
        for idx in self.batch_idxs:
            # print("batch:\t\t", *self.imgs_by_classes[idx])
            assert \
                sum(
                    map(
                        lambda file_path: int(get_class_name(file_path)),
                        self.imgs_by_classes[idx]
                    )
                ) / batch_size == int(get_class_name(self.imgs_by_classes[idx][0])), \
                "Wrong batch size: You should use a butch size less than (or equal) 16, and a multiple of 2."
            # ToDo - минимальный размер должен быть 140 (receptive field)

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        idx = self.batch_idxs[idx]
        self.files_names_batch = self.imgs_by_classes[idx]
        # ToDo
        ''' 
            Сейчас на вход будет подаваться 1 3D объем из batch_size снимков, а нужно несколько.
            Группировка (by reshape) должна происходить на уровне __init__
        '''
        fimages1c = np.zeros((1, self.img_size[0], self.img_size[1], self.batch_size, 1), dtype=np.float32)
        fmasks1c = np.zeros((1, self.img_size[0], self.img_size[1], self.batch_size, 1), dtype=np.float32)

        for jdx, file_name in zip(range(self.batch_size), self.files_names_batch):
            img_path = join(self.imgs_patch, file_name)
            mask_path = join(self.masks_patch, file_name)
            fimage1с = tf.image.convert_image_dtype(
                tf.io.decode_jpeg(tf.io.read_file(img_path), channels=self.channels),
                tf.float32
            )
            fmask1с = tf.image.convert_image_dtype(
                tf.io.decode_jpeg(tf.io.read_file(mask_path), channels=self.channels),
                tf.float32
            )

            fimages1c[0, :, :, jdx, :] = fimage1с
            fmasks1c[0, :, :, jdx, :] = fmask1с

        fmasks1c = tf.where(condition=tf.math.greater(fmasks1c, 0.5), x=1.0, y=0.0)

        return fimages1c, fmasks1c
        '''
        # ToDo - Возможно, эта реализация работает быстрее, протестировать
        img_path = join(self.imgs_patch, file_name)
        mask_path = join(self.masks_patch, file_name)
        fimages1с = tf.expand_dims(
            tf.image.convert_image_dtype(
                tf.io.decode_jpeg(tf.io.read_file(img_path), channels=1),
                tf.float32
            ), axis=2
        )
        fmasks1с = tf.expand_dims(
            tf.image.convert_image_dtype(
                tf.io.decode_jpeg(tf.io.read_file(mask_path), channels=1),
                tf.float32
            ), axis=2
        )
        print(fimages1с.shape, fmasks1с.shape)

        for file_name in self.files_names_batch[1:]:
            img_path = join(self.imgs_patch, file_name)
            mask_path = join(self.masks_patch, file_name)
            fimages1с = tf.concat(
                [fimages1с,
                tf.expand_dims(
                    tf.image.convert_image_dtype(
                        tf.io.decode_jpeg(tf.io.read_file(img_path), channels=1),
                        tf.float32
                    ), axis=2
                )], axis=2
            )
            fmasks1с = tf.concat(
                [fmasks1с,
                 tf.expand_dims(
                     tf.image.convert_image_dtype(
                        tf.io.decode_jpeg(tf.io.read_file(mask_path), channels=1),
                        tf.float32
                     ), axis=2
                )], axis=2
            )

        print(fimages1с.shape, fmasks1с.shape)
        fimages1с = tf.expand_dims(fimages1с, axis=0)
        fmasks1с = tf.expand_dims(fmasks1с, axis=0)
        print(fimages1с.shape, fmasks1с.shape)
        fmasks1c = tf.where(condition=tf.math.greater(fmasks1c, 0.5), x=1.0, y=0.0)

        return fimages1с, fmasks1с
        '''
