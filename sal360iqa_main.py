import numpy as np
import tensorflow as tf
import random as python_random
from scipy.stats import pearsonr, spearmanr
import os
import argparse
import pandas as pd
import csv
import model
import dataset
from sklearn.model_selection import train_test_split

os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

print(tf.__version__)

SEED = 123
np.random.seed(SEED)
python_random.seed(SEED)
tf.random.set_seed(SEED)

NBR_PATCHES = 64
OIQA_IMG = 320



def select_gpu(id_gpu):
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[int(id_gpu)], 'GPU')


if __name__ == '__main__':

    EPOCHS = 100  # could be more or less

    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", "--bs",
                        help="batch size.", type=int)
    parser.add_argument("-db", "--db",
                        help="database name.")
    parser.add_argument("-gpu", "--gpu_id",
                        help="GPU ID to be used.")
    parser.add_argument("-val", "--val", type=int,
                        help="Validation percentage.")
    parser.add_argument("-norm", "--norm", type=int,
                        help="Wether to use normalization or not.")
    parser.add_argument("-loss", "--ls",
                        help="The loss function")

    args = parser.parse_args()

    batch_size = args.bs
    database = args.db
    gpu_id = args.gpu_id
    val = float(args.val/10)
    normalization = args.norm
    loss = args.ls

    select_gpu(gpu_id)

    out_p = os.path.join('Results/', database)
    os.makedirs(out_p, exist_ok=True)

    opt = tf.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-4 / EPOCHS)

    if loss == 'huber':
        ls_func = tf.keras.losses.Huber()
    if loss == 'mse':
        ls_func = tf.keras.losses.MeanSquaredError()
    if loss == 'mae':
        ls_func = tf.keras.losses.MeanAbsoluteError()

    if normalization == 1:
        inp_s = (128, 128, 1)
        pre_norm = 'LCN'
    elif normalization == 0:
        inp_s = (256, 256, 3)
        pre_norm = 'RGB'

    # Read your data
    data = dataset.Dataset()
    patches, mos = data.get_input(
        patches_data, patches_path, OIQA_IMG, NBR_PATCHES, normalization)

    # Split into training and testing sets (train_x, train_y) (test_x, test_y)
    train_x, test_x, train_y, test_y = train_test_split(
        patches, mos, test_size=0.2, random_state=42)

    sal360iqa = model.Sal360Model()
    iqa_model = Sal360Model.build_model(inp_s, out_dim)

    print('[INFO] Compiling the model...')
    iqa_model.compile(loss=ls_func,
                      optimizer=opt, metrics=tf.keras.metrics.RootMeanSquaredError(name='rmse'))

    cb = model.create_callbacks_fun(
        1, out_p, batch_size, val, pre_norm, database)

    print('[INFO] Training the model...')
    iqa_model.fit(x=train_x, y=train_y, validation_split=val,
                  epochs=EPOCHS, batch_size=batch_size, callbacks=cb, shuffle=True)

    preds = iqa_model.predict(test_x, batch_size=batch_size)

    plcc = pearsonr(preds, test_y)
    srocc = spearmanr(preds, test_y)

    print(f'PLCC = {plcc[0]}, SRCC = {srocc[0]}')
