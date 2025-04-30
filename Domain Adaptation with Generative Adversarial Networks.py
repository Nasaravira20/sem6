import tensorflow as tf
from keras import layers, models, optimizers
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Define the Domain Adaptation GAN
class DomainAdaptationGAN:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.adversarial_model = self.build_adversarial_model()

    def build_generator(self):
        model = models.Sequential([
            layers.InputLayer(input_shape=self.input_shape),
            layers.Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), strides=2, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2DTranspose(128, (3, 3), strides=2, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(3, (3, 3), strides=1, padding='same', activation='tanh'),
        ])
        return model

    def build_discriminator(self):
        model = models.Sequential([
            layers.InputLayer(input_shape=self.input_shape),
            layers.Conv2D(64, (3, 3), strides=2, padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), strides=2, padding='same'),
            layers.LeakyReLU(alpha=0.2),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dense(1, activation='sigmoid'),
        ])
        model.compile(optimizer=optimizers.Adam(0.0002, beta_1=0.5),
                      loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def build_adversarial_model(self):
        self.discriminator.trainable = False
        model = models.Sequential([
            self.generator,
            self.discriminator
        ])
        model.compile(optimizer=optimizers.Adam(0.0002, beta_1=0.5),
                      loss='binary_crossentropy')
        return model

    def train(self, source_data, target_data, epochs, batch_size):
        for epoch in range(epochs):
            # Train discriminator with source and target samples
            idx_s = tf.convert_to_tensor(np.random.randint(0, source_data.shape[0], batch_size // 2), dtype=tf.int32)
            idx_t = tf.convert_to_tensor(np.random.randint(0, target_data.shape[0], batch_size // 2), dtype=tf.int32)

            real_source = tf.gather(source_data, idx_s)
            real_target = tf.gather(target_data, idx_t)

            fake_target = self.generator.predict(real_source)

            real_labels = np.ones((batch_size // 2, 1))
            fake_labels = np.zeros((batch_size // 2, 1))

            d_loss_real = self.discriminator.train_on_batch(real_target, real_labels)
            d_loss_fake = self.discriminator.train_on_batch(fake_target, fake_labels)

            # Train generator to fool discriminator
            g_loss = self.adversarial_model.train_on_batch(real_source, real_labels)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, D Loss: {0.5 * np.add(d_loss_real, d_loss_fake)}, G Loss: {g_loss}")

# Prepare Source and Target Data (Example data)
def preprocess_data(data, input_shape):
    data = tf.image.resize(data, (input_shape[0], input_shape[1]))
    return data / 127.5 - 1.0

# Example usage
input_shape = (64, 64, 3)
source_data = np.random.random((1000, 64, 64, 3))  # Replace with actual source data
target_data = np.random.random((1000, 64, 64, 3))  # Replace with actual target data

source_data = preprocess_data(source_data, input_shape)
target_data = preprocess_data(target_data, input_shape)

# Initialize and train the GAN
adaptation_gan = DomainAdaptationGAN(input_shape)
adaptation_gan.train(source_data, target_data, epochs=100, batch_size=32)

# Evaluate on target domain tasks (Custom evaluation logic here)
print("Domain adaptation completed.")
