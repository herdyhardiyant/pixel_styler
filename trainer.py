import losses
import tensorflow as tf
import features_extractor

def calculate_gradients(model: tf.keras.Model,
                        generated_image,
                        content_targets, style_targets,
                        style_weight,
                        content_weight,
                        num_style_layers,
                        var_weight = 2
                        ):
    with tf.GradientTape() as tape:
        generated_style_features = features_extractor.get_style_image_features(model, generated_image, num_style_layers)
        generated_content_features = features_extractor.get_content_image_features(model, generated_image)
        loss = losses.total_loss(generated_content_features,
                                 generated_style_features,
                                 content_targets,
                                 style_targets,
                                 style_weight,
                                 content_weight,
                                 num_style_layers)

        loss += var_weight * tf.image.total_variation(generated_image)

    gradients = tape.gradient(loss, generated_image)
    return gradients
