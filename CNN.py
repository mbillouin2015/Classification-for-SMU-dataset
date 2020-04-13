import plaidml.keras
plaidml.keras.install_backend()
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import seaborn as sns

NDEBUG = 1
if NDEBUG:
    DATADIR = 'data/FordA.pickle'
    # SAVEDIR = 'plots/'

    data = pd.read_pickle(DATADIR)

    x_train = data[data['split'] == 'TRAIN'].ts
    x_train = np.array([i for i in x_train])

    y_train = pd.get_dummies(data[data['split'] == 'TRAIN'].class_lbl).values

    x_test = data[data['split'] == 'TEST'].ts
    x_test = np.array([i for i in x_test])
    y_test = pd.get_dummies(data[data['split'] == 'TEST'].class_lbl).values

else:
    DATADIR = 'data/smu_resampled_balanced.pickle'
    data = pd.read_pickle(DATADIR)

    x_train, x_test, y_train, y_test = train_test_split(data, data['lbl'], train_size=0.70, random_state=42)

    x_train = x_train.ts.values
    x_train = np.array([i for i in x_train])

    y_train = pd.get_dummies(y_train).values

    x_test = x_test.ts.values
    x_test = np.array([i for i in x_test])

    y_test = pd.get_dummies(y_test).values

