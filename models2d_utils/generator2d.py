from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


def prepare_images2d_batch(images_batch):
    return images_batch / 255.0


def prepare_masks2d_batch(masks_batch):
    masks_batch = masks_batch / 255.0
    masks_batch = tf.where(condition=tf.math.greater(masks_batch, 0.5), x=1.0, y=0.0)
    return masks_batch


def get_image_mask_generator2d(
        main_dir_path, imgs_dir_name, masks_dir_name,
        color_mode="grayscale", target_size=(512, 512),
        interpolation="bilinear", aug_dict={},
        batch_size=4, shuffle=True, seed=7,
):
    images_datagen = ImageDataGenerator(**aug_dict)
    masks_datagen = ImageDataGenerator(**aug_dict)

    images_generator = images_datagen.flow_from_directory(
        main_dir_path,
        target_size=target_size,
        color_mode=color_mode,
        classes=[imgs_dir_name],
        class_mode=None,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        interpolation=interpolation,
        # ToDo - label_mode=None, #Float32
    )
    masks_generator = masks_datagen.flow_from_directory(
        main_dir_path,
        target_size=target_size,
        color_mode=color_mode,
        classes=[masks_dir_name],
        class_mode=None,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        interpolation=interpolation,
        # ToDo - label_mode=None, #Float32
    )

    images_masks_generator = zip(images_generator, masks_generator)

    for (images_batch, masks_batch) in images_masks_generator:
        images_batch = prepare_images2d_batch(images_batch)
        masks_batch = prepare_masks2d_batch(masks_batch)
        yield (images_batch, masks_batch)


def get_image_generator2d(
        main_dir_path, imgs_dir_name,
        color_mode="grayscale", target_size=(512, 512),
        interpolation='bilinear',
        batch_size=4,
):
    images_datagen = ImageDataGenerator(**dict())
    images_generator = images_datagen.flow_from_directory(
        main_dir_path,
        target_size=target_size,
        color_mode=color_mode,
        classes=[imgs_dir_name],
        class_mode=None,
        batch_size=batch_size,
        shuffle=False,
        interpolation=interpolation,
    )

    for images_batch in images_generator:
        images_batch = prepare_images2d_batch(images_batch)
        yield images_batch
