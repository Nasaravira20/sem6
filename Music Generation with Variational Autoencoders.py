import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers
import numpy as np
import pretty_midi
import glob
import os
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.losses import MeanSquaredError

# Use CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Define constants
LATENT_DIM = 128
INPUT_DIM = 128  # MIDI pianoroll representation
SEQ_LENGTH = 100
BATCH_SIZE = 64
EPOCHS = 50

# Preprocessing function
def preprocess_midi(file_paths, seq_length):
    data = []
    for file_path in file_paths:
        try:
            midi_data = pretty_midi.PrettyMIDI(file_path)
            piano_roll = midi_data.get_piano_roll(fs=10)
            for i in range(0, piano_roll.shape[1] - seq_length, seq_length):
                data.append(piano_roll[:, i:i + seq_length].T)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    data = np.array(data) / 127.0
    return data.reshape(-1, seq_length * INPUT_DIM)

# Encoder class
class Encoder(layers.Layer):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.dense1 = layers.Dense(256, activation='relu')
        self.dense2 = layers.Dense(128, activation='relu')
        self.z_mean = layers.Dense(latent_dim)
        self.z_log_var = layers.Dense(latent_dim)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        return z_mean, z_log_var

# Decoder class
class Decoder(layers.Layer):
    def __init__(self, output_dim):
        super(Decoder, self).__init__()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(256, activation='relu')
        self.dense3 = layers.Dense(output_dim, activation='sigmoid')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)

# VAE class
class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        reconstructed = self.decoder(z)
        kl_loss = -0.5 * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1
        )
        self.add_loss(tf.reduce_mean(kl_loss))
        return reconstructed

# Train the VAE
def build_and_train_vae(data, latent_dim, input_dim, batch_size, epochs):
    print(f"Dataset shape: {data.shape}")
    encoder = Encoder(latent_dim)
    decoder = Decoder(input_dim)
    vae = VAE(encoder, decoder)

    vae.compile(optimizer='adam', loss=MeanSquaredError())
    vae.fit(data, data, batch_size=batch_size, epochs=epochs)

    return vae


# Sample from latent space and generate music
def generate_music(vae, num_samples, seq_length):
    latent_samples = tf.random.normal(shape=(num_samples, LATENT_DIM))
    generated_sequences = vae.decoder(latent_samples)
    return generated_sequences.numpy().reshape(num_samples, seq_length, INPUT_DIM)

# === Main Execution ===
if __name__ == "__main__":
    file_paths = glob.glob("audio/*.mid")
    data = preprocess_midi(file_paths, SEQ_LENGTH)
    vae = build_and_train_vae(data, LATENT_DIM, INPUT_DIM * SEQ_LENGTH, BATCH_SIZE, EPOCHS)

    print("Generating new music samples...")
    generated_music = generate_music(vae, num_samples=10, seq_length=SEQ_LENGTH)
    