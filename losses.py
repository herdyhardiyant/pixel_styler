import tensorflow as tf


def content_loss(output, target):
    return 0.5 * tf.reduce_sum(tf.square(output - target))


def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations


def style_loss(style, target):
    return tf.reduce_mean(tf.square(gram_matrix(style) - gram_matrix(target)))


def total_loss(content_outputs, style_outputs, content_targets, style_targets, style_weight, content_weight, num_layers):
    content_loss_value = tf.add_n([content_loss(output, target)
                                   for output, target in zip(content_outputs, content_targets)])

    style_loss_value = tf.add_n([style_loss(style_output, style_target)
                                 for style_output, style_target in zip(style_outputs, style_targets)])

    content_loss_value = content_loss_value * content_weight / num_layers
    style_loss_value = style_loss_value * style_weight

    total_loss = content_loss_value + style_loss_value
    return total_loss
