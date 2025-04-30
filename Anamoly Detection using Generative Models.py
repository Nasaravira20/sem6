import tensorflow as tf
from keras import layers, models, losses, optimizers, datasets
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Define the VAE Architecture
class VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = models.Sequential([
            layers.InputLayer(input_shape=(28, 28, 1)),
            layers.Conv2D(32, (3, 3), activation="relu", strides=2, padding="same"),
            layers.Conv2D(64, (3, 3), activation="relu", strides=2, padding="same"),
            layers.Flatten(),
            layers.Dense(latent_dim + latent_dim),  # Mean and log-variance
        ])

        # Decoder
        self.decoder = models.Sequential([
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Dense(7 * 7 * 64, activation="relu"),
            layers.Reshape((7, 7, 64)),
            layers.Conv2DTranspose(64, (3, 3), activation="relu", strides=2, padding="same"),
            layers.Conv2DTranspose(32, (3, 3), activation="relu", strides=2, padding="same"),
            layers.Conv2DTranspose(1, (3, 3), activation="sigmoid", padding="same"),
        ])

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        return logits

    def call(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z)

# Loss function
@tf.function
def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    reconstruction_loss = tf.reduce_mean(
        tf.reduce_sum(losses.binary_crossentropy(x, x_logit), axis=(1, 2))
    )
    kl_divergence = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar))
    return reconstruction_loss + kl_divergence

# Training step
@tf.function
def train_step(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# Load and preprocess dataset
(x_train, _), (x_test, _) = datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Train VAE
latent_dim = 2
vae = VAE(latent_dim=latent_dim)
optimizer = optimizers.Adam(learning_rate=1e-3)
epochs = 20
batch_size = 64

dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(batch_size)
for epoch in range(epochs):
    for batch in dataset:
        train_step(vae, batch, optimizer)
    print(f"Epoch {epoch + 1}/{epochs} completed.")

# Anomaly Detection
reconstruction_errors = []
for x in x_test:
    x = tf.expand_dims(x, axis=0)
    reconstruction = vae(x)
    error = tf.reduce_mean(tf.abs(x - reconstruction))
    reconstruction_errors.append(error.numpy())

# Define threshold
threshold = np.percentile(reconstruction_errors, 95)

# Evaluate anomalies
def detect_anomalies(data, model, threshold):
    anomalies = []
    for x in data:
        x = tf.expand_dims(x, axis=0)
        reconstruction = model(x)
        error = tf.reduce_mean(tf.abs(x - reconstruction))
        if error > threshold:
            anomalies.append(True)
        else:
            anomalies.append(False)
    return anomalies

# Example usage
anomalies = detect_anomalies(x_test, vae, threshold)
print(f"Detected anomalies: {np.sum(anomalies)} out of {len(x_test)} samples.")
