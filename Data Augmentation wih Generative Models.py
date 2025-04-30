import tensorflow as tf
from keras import layers, models, datasets, utils, optimizers
import numpy as np
from sklearn.model_selection import train_test_split
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Load a sample dataset (e.g., MNIST for demonstration)
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)

# Split into training and validation
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Define a simple GAN for data generation
class GAN:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = self.build_gan()

    def build_generator(self):
        model = models.Sequential([
            layers.Dense(256, activation='relu', input_dim=self.input_dim),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(28 * 28, activation='sigmoid'),
            layers.Reshape((28, 28, 1))
        ])
        return model

    def build_discriminator(self):
        model = models.Sequential([
            layers.Flatten(input_shape=(28, 28, 1)),
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=optimizers.Adam(0.0002),
                      loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def build_gan(self):
        self.discriminator.trainable = False
        model = models.Sequential([
            self.generator,
            self.discriminator
        ])
        model.compile(optimizer=optimizers.Adam(0.0002),
                      loss='binary_crossentropy')
        return model

    def train(self, x_train, epochs, batch_size):
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # Train discriminator
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            real_images = x_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.input_dim))
            generated_images = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(real_images, real_labels)
            d_loss_fake = self.discriminator.train_on_batch(generated_images, fake_labels)

            # Train generator
            noise = np.random.normal(0, 1, (batch_size, self.input_dim))
            g_loss = self.gan.train_on_batch(noise, real_labels)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, D Loss: {0.5 * np.add(d_loss_real, d_loss_fake)}, G Loss: {g_loss}")

# Train GAN
gan = GAN(input_dim=100)
gan.train(x_train, epochs=500, batch_size=64)

# Generate augmented data
num_generated = 10000
noise = np.random.normal(0, 1, (num_generated, 100))
generated_images = gan.generator.predict(noise)
generated_labels = utils.to_categorical(np.random.randint(0, 10, num_generated), 10)

x_augmented = np.concatenate((x_train, generated_images), axis=0)
y_augmented = np.concatenate((y_train, generated_labels), axis=0)

# Define a CNN classifier
def build_classifier():
    model = models.Sequential([
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train classifiers
classifier_original = build_classifier()
classifier_augmented = build_classifier()

print("Training on original data...")
classifier_original.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=64)

print("Training on augmented data...")
classifier_augmented.fit(x_augmented, y_augmented, validation_data=(x_val, y_val), epochs=10, batch_size=64)

# Evaluate classifiers
original_acc = classifier_original.evaluate(x_test, y_test, verbose=0)[1]
augmented_acc = classifier_augmented.evaluate(x_test, y_test, verbose=0)[1]

print(f"Accuracy on original data: {original_acc:.2f}")
print(f"Accuracy with augmented data: {augmented_acc:.2f}")
