# from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf


def preprocess_image(image_path:str):
    max_dim = 512
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)

    shape = tf.shape(image)[:-1]
    shape = tf.cast(shape, tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    image = tf.image.resize(image, new_shape)
    image = image[tf.newaxis, :]
    image = tf.image.convert_image_dtype(image, tf.uint8)

    image = tf.cast(image, dtype=tf.float32)
    image = tf.keras.applications.vgg19.preprocess_input(image)

    #
    # img = image.load_img(image_path, target_size=(512, 512))
    # img = image.img_to_array(img)
    # img = np.expand_dims(img, axis=0)
    # img = tf.keras.applications.vgg19.preprocess_input(img)

    return image
