import numpy as np
import tensorflow as tf

initializer = tf.keras.initializers.he_normal()

class Sal360Model():
    def __init__(self):
        pass

    def regression_block(self, features, out_dim):

        x = tf.keras.layers.Dense(1024, activation="relu",
                                kernel_initializer=initializer)(features)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(512, activation="relu",
                                kernel_initializer=initializer)(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        x = tf.keras.layers.Dense(
            out_dim, activation='linear', kernel_initializer=initializer)(x)

        return x
    
    def build_model(self, input_size, out_dim, filters=(64, 128, 256, 512)):
        
        inputs = tf.keras.layers.Input(shape=input_size)

        for (i, f) in enumerate(filters):
            if i == 0:
                x = inputs
            x = tf.keras.layers.Conv2D(filters=f, kernel_size=(
                3, 3), padding="same", kernel_initializer=initializer)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)

            x = tf.keras.layers.Conv2D(filters=f, kernel_size=(
                3, 3), padding="same", kernel_initializer=initializer)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)

            max_p = tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2), strides=(2, 2))(x)
            avrg_p = tf.keras.layers.AveragePooling2D(
                pool_size=(2, 2), strides=(2, 2))(x)
            min_p = -tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2), strides=(2, 2))(-x)

            x = tf.keras.layers.concatenate([avrg_p, max_p, min_p])

        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        output = self.regression_block(x, out_dim)

        iqa_model = tf.keras.models.Model(inputs=inputs, outputs=output)
        return iqa_model
