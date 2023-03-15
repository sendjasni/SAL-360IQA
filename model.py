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

    class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

    def __init__(self, patience=15):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


def create_callbacks_fun(fold_no, path, bs, version):

    log_file = path + '/Losses_fold_' + str(fold_no) + \
        '_BS_' + str(bs) + '_version_' + version + '.log'

    csv_loger = tf.keras.callbacks.CSVLogger(
        log_file, separator=",", append=False)

    filepath = path + '/Weights_fold_' + str(fold_no) + \
        '_BS_' + str(bs) + '_version_' + version + '.hdf5'

    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath,
                                                    monitor="val_loss",
                                                    mode="min",
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    save_freq='epoch',
                                                    verbose=1)

    callbacks = [
        EarlyStoppingAtMinLoss(),
        csv_loger,
        checkpoint,
    ]

    return callbacks
