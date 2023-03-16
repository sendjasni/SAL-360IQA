import numpy as np
import tensorflow as tf
import random as python_random
from scipy.stats import pearsonr, spearmanr
import os
import argparse
import pandas as pd
import csv
import model


os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

print(tf.__version__)

SEED = 123
np.random.seed(SEED)
python_random.seed(SEED)
tf.random.set_seed(SEED)
