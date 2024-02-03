import tensorflow as tf
import process

def get_model(content_layers, style_layers):
    vgg19 = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg19.trainable = False
    content_outputs = [vgg19.get_layer(layer).output for layer in content_layers]
    style_outputs = [vgg19.get_layer(layer).output for layer in style_layers]
    model_outputs = style_outputs + content_outputs
    model = tf.keras.Model(vgg19.input, model_outputs)

    for layer in model.layers:
        layer.trainable = False

    return model


def get_style_image_features(model: tf.keras.Model, preprocessed_image, num_styles_layer: int):
    outputs = model(preprocessed_image)
    style_outputs = outputs[:num_styles_layer]
    return style_outputs

def get_content_image_features(model: tf.keras.Model, preprocessed_image):
    outputs = model(preprocessed_image)
    return outputs[-1]


