from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf


def preprocess_image(image_path:str):
    img = image.load_img(image_path, target_size=(512, 512))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img
