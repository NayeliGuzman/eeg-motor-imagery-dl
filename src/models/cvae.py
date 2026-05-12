# -----------------
# Packages
# -----------------
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras import optimizers


# -----------------
# Sampling Layer — reparameterization trick
# epsilon -> random noise, exp(0.5 * z_log_var) -> converts log variance to standard deviation
# rerurns sampled latent vector
# -----------------
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# -----------------
# Encoder
# -----------------
def build_encoder(input_shape, latent_dim):
# '''z-mean: center of latent distribution - most stable, deterministic representation
#     z_log_var: variance
#     z: sampled latent vector
# '''
    encoder_inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, activation='relu', padding='same')(encoder_inputs) #for a smaller model
    x = layers.MaxPooling2D(2, padding='same')(x)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x) 
    x = layers.MaxPooling2D(2, padding='same')(x)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2, padding='same')(x)

    print("the output encoder shape of x is ", x.shape)
    shape_before_flatten = tf.keras.backend.int_shape(x)[1:]
    x = layers.Flatten()(x)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])
    return tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder'), shape_before_flatten


# -----------------
# Decoder
# -----------------
def build_decoder(latent_dim, shape_before_flatten):
    latent_inputs = tf.keras.Input(shape=(latent_dim,))
    print(type(shape_before_flatten))
    print(shape_before_flatten)
    units = int(np.prod(shape_before_flatten))

    x = layers.Dense(units, activation='relu')(latent_inputs)
    x = layers.Reshape(shape_before_flatten)(x)

    # for a larger model
    # x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
    # x = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
    # x = layers.Conv2DTranspose(16, 3, strides=2, padding='same', activation='relu')(x)

    # for a smaller model
    x = layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(16, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(8, 3, strides=2, padding='same', activation='relu')(x)

    x = layers.Conv2DTranspose(1, 3, padding='same')(x)

    # ensures exact match with input
    decoder_outputs = layers.Resizing(22, 250)(x)

    return tf.keras.Model(latent_inputs, decoder_outputs, name='decoder')

# -----------------
#CVAE w/ modifiable KL weight
# -----------------
class CVAE(tf.keras.Model):
    def __init__(self, encoder, decoder, kl_weight=0.0001):
        super(CVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.kl_weight = kl_weight

    # call(); REQUIRED by Keras
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z) #reconstruction

    def train_step(self, data):
        x = data

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)

            # Reconstruction loss
            recon_loss = tf.reduce_mean(
                tf.keras.losses.mse(x, reconstruction)
            )

            kl_loss = -0.5 * tf.reduce_mean(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))

            # Combine losses
            total_loss = recon_loss + self.kl_weight * kl_loss
        tf.debugging.check_numerics(total_loss, "loss is NAN or inf!")
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {
            "loss": total_loss,
            "reconstruction_loss": recon_loss,
            "kl_loss": kl_loss,
        }
