import tensorflow as tf
from keras import layers, models
import numpy as np
import os
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Define 3D GAN for Video Generation
class VideoGAN:
    def __init__(self, noise_dim, video_shape):
        self.noise_dim = noise_dim
        self.video_shape = video_shape
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = self.build_gan()

    def build_generator(self):
        model = models.Sequential([
            layers.Dense(4 * 4 * 4 * 512, activation="relu", input_dim=self.noise_dim),
            layers.Reshape((4, 4, 4, 512)),  # Start with a small 3D volume
            layers.Conv3DTranspose(256, kernel_size=(4, 4, 4), strides=(2, 2, 2),
                                padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv3DTranspose(128, kernel_size=(4, 4, 4), strides=(2, 2, 2),
                                padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv3DTranspose(64, kernel_size=(4, 4, 4), strides=(2, 2, 2),
                                padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv3DTranspose(3, kernel_size=(4, 4, 4), strides=(1, 2, 2),
                                padding="same", activation="tanh")  # Final shape: (32, 64, 64, 3)
        ])
        return model

    def build_discriminator(self):
        model = models.Sequential([
            layers.Conv3D(64, kernel_size=(4, 4, 4), strides=(2, 2, 2), padding="same",
                          input_shape=self.video_shape),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv3D(128, kernel_size=(4, 4, 4), strides=(2, 2, 2), padding="same"),
            layers.LeakyReLU(alpha=0.2),
            layers.Flatten(),
            layers.Dense(1, activation="sigmoid")
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
                      loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def build_gan(self):
        self.discriminator.trainable = False
        model = models.Sequential([
            self.generator,
            self.discriminator
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
                      loss="binary_crossentropy")
        return model

    def train(self, video_data, epochs, batch_size):
        half_batch = batch_size // 2

        for epoch in range(epochs):
            # Train Discriminator
            idx = np.random.randint(0, video_data.shape[0], half_batch)
            real_videos = video_data[idx]
            noise = np.random.normal(0, 1, (half_batch, self.noise_dim))
            generated_videos = self.generator.predict(noise)

            real_labels = np.ones((half_batch, 1))
            fake_labels = np.zeros((half_batch, 1))

            d_loss_real = self.discriminator.train_on_batch(real_videos, real_labels)
            d_loss_fake = self.discriminator.train_on_batch(generated_videos, fake_labels)

            # Train Generator
            noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
            valid_labels = np.ones((batch_size, 1))
            g_loss = self.gan.train_on_batch(noise, valid_labels)

            # Print losses every epoch
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, D Loss: {0.5 * np.add(d_loss_real, d_loss_fake)}, G Loss: {g_loss}")

# Preprocess Video Data (Example assumes dataset of shape [num_samples, time_steps, height, width, channels])
def preprocess_videos(video_paths, video_shape):
    videos = []
    for path in video_paths:
        video = np.random.random(video_shape)  # Replace with video loading logic
        videos.append(video)
    return np.array(videos, dtype="float32") / 127.5 - 1.0

# Example Usage
video_shape = (32, 64, 64, 3)  # (time_steps, height, width, channels)
noise_dim = 100
video_paths = ["videos/elephant.mp4", "videos/giraffes.mp4"]  # Replace with actual video file paths
video_data = preprocess_videos(video_paths, video_shape)

video_gan = VideoGAN(noise_dim=noise_dim, video_shape=video_shape)
video_gan.train(video_data, epochs=20, batch_size=16)

# Generate and save video sequences
num_videos = 5
noise = np.random.normal(0, 1, (num_videos, noise_dim))
generated_videos = video_gan.generator.predict(noise)

# Save generated videos
for i, video in enumerate(generated_videos):
    save_path = f"generated_video_{i}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, 10, (video.shape[2], video.shape[1]))

    # Normalize video frames to [0, 255] and save each frame
    for frame in video:
        frame = ((frame + 1) * 127.5).astype(np.uint8)  # Rescale to [0, 255]
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR for OpenCV

    out.release()
    print(f"Generated video saved to {save_path}")
