import tensorflow as tf
import features_extractor
import process
import losses
import trainer
from IPython.display import Image, clear_output
from IPython.display import display
import visualizer

content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

num_style_layers = len(style_layers)
num_content_layers = len(content_layers)

content_path = "./images/cat.jpg"
style_path = "./images/tnj.jpg"

content_weight = 1e-2
style_weight = 2e-2
loss_weights = {'content': content_weight, 'style': style_weight}

model = features_extractor.get_model(content_layers, style_layers)

content_image = process.preprocess_image(content_path)
style_image = process.preprocess_image(style_path)

content_targets = features_extractor.get_content_image_features(model, content_image)
style_targets = features_extractor.get_style_image_features(model, style_image, num_style_layers)

generated_image = tf.cast(content_image, dtype=tf.float32)
generated_image = tf.Variable(generated_image)
optimizer = tf.keras.optimizers.Adam(
    tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=20.0, decay_steps=100, decay_rate=0.50
    )
)

images = []
images.append(generated_image)
epochs = 10
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    for step in range(100):

        gradients = trainer.calculate_gradients(
            model,
            generated_image,
            content_targets,
            style_targets,
            content_weight,
            style_weight,
            num_style_layers)
        optimizer.apply_gradients([(gradients, generated_image)])
        clipped_image = tf.clip_by_value(generated_image, 0.0, 255.0)
        generated_image.assign(clipped_image)
        print(".", end='')
        if (step + 1) % 10 == 0:
            images.append(generated_image)
            print("\r", end='')

clear_output(wait=True)
generated_image = tf.cast(generated_image, dtype=tf.uint8)
display_image = visualizer.tensor_to_image(images[-1])
display_image.show()
