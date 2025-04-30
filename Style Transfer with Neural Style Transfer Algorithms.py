import tensorflow as tf
from keras.src.applications.vgg19 import VGG19, preprocess_input
from keras.src.models import Model
import numpy as np
from PIL import Image
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Aim: Implement Neural Style Transfer to blend the style of one image into another.

def load_and_preprocess_image(image_path, target_dim):
    img = Image.open(image_path)
    img = img.resize((target_dim, target_dim))
    img_array = np.array(img, dtype=np.float32)
    # Preprocess for VGG19: BGR, zero-centered
    img_pre = preprocess_input(img_array)
    return tf.expand_dims(img_pre, axis=0)


def postprocess_image(img_tensor):
    # Remove batch dimension
    img = img_tensor[0].numpy()
    # Add back mean, clip, convert from BGR to RGB
    img = img + [103.939, 116.779, 123.68]
    img = np.clip(img, 0, 255).astype('uint8')
    img = img[..., ::-1]
    return Image.fromarray(img)


def get_vgg_model(style_layers, content_layers):
    vgg = VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in (style_layers + content_layers)]
    model = Model(inputs=vgg.input, outputs=outputs)
    return model


def gram_matrix(tensor):
    # Compute Gram matrix for style representation
    x = tf.squeeze(tensor, axis=0)
    features = tf.reshape(x, [-1, x.shape[-1]])
    gram = tf.matmul(features, features, transpose_a=True)
    return gram / tf.cast(tf.shape(features)[0], tf.float32)


def compute_loss(model, loss_weights, init_image, gram_style_features, content_features, style_layers, content_layers):
    style_weight, content_weight = loss_weights
    # Get model outputs
    outputs = model(init_image)
    style_outputs = outputs[:len(style_layers)]
    content_outputs = outputs[len(style_layers):]

    # Compute style loss
    s_loss = tf.add_n([
        tf.reduce_mean((gram_matrix(style_out) - gram_target) ** 2)
        for style_out, gram_target in zip(style_outputs, gram_style_features)
    ])
    s_loss *= style_weight / len(style_layers)

    # Compute content loss
    c_loss = tf.add_n([
        tf.reduce_mean((content_out - content_target) ** 2)
        for content_out, content_target in zip(content_outputs, content_features)
    ])
    c_loss *= content_weight / len(content_layers)

    return s_loss + c_loss


def style_transfer(
    content_path,
    style_path,
    target_dim=512,
    iterations=1000,
    style_weight=1e-2,
    content_weight=1e4
):
    # Load and preprocess images
    content_image = load_and_preprocess_image(content_path, target_dim)
    style_image = load_and_preprocess_image(style_path, target_dim)

    # Define layers to extract
    style_layers = [
        'block1_conv1', 'block2_conv1',
        'block3_conv1', 'block4_conv1', 'block5_conv1'
    ]
    content_layers = ['block5_conv2']

    # Build the model
    model = get_vgg_model(style_layers, content_layers)

    # Extract style and content features
    style_outputs = model(style_image)[:len(style_layers)]
    content_outputs = model(content_image)[len(style_layers):]

    gram_style_features = [gram_matrix(style_output) for style_output in style_outputs]
    content_features = content_outputs

    # Initialize generated image
    init_image = tf.Variable(content_image, dtype=tf.float32)

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)

    # Training loop
    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            loss = compute_loss(
                model,
                (style_weight, content_weight),
                init_image,
                gram_style_features,
                content_features,
                style_layers,
                content_layers
            )
        grad = tape.gradient(loss, init_image)
        optimizer.apply_gradients([(grad, init_image)])
        # Clip to maintain valid pixel range for VGG preprocessing
        init_image.assign(tf.clip_by_value(init_image, -103.939, 255 - 103.939))
        return loss

    for i in range(1, iterations + 1):
        loss = train_step()
        if i % 100 == 0:
            print(f"Iteration {i}/{iterations}, loss: {loss:.4e}")

    # Return the stylized image
    return postprocess_image(init_image)

# Example Usage
if __name__ == '__main__':
    content_path = 'images/image1.png'
    style_path = 'images/image.png'
    output = style_transfer(
        content_path,
        style_path,
        iterations=500,
        style_weight=1e-2,
        content_weight=1e4,
        target_dim=512
    )
    output.show()
