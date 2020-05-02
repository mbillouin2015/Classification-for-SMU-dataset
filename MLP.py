import time
import plaidml.keras
plaidml.keras.install_backend()
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import numpy as np
from numba import jit
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from keras import backend as K
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from generateSpectrogram import generateSpectrogramFeatures


import seaborn as sns

RESULTSDIR = 'results/MLP/'
NDEBUG = 0

if NDEBUG:
    RESULTSDIR += 'baseline/'
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
    RESULTSDIR += 'smu/'
    DATADIR = 'data/smu_resampled_balanced.pickle'
    data = pd.read_pickle(DATADIR)

    float_epsilon = np.finfo(float).eps
    data['ts'] = data.ts.map(lambda x: (x-x.mean())/(x.std()+float_epsilon))

    # generateSpectrogramFeatures(data)
    x_train, x_test, y_train, y_test = train_test_split(data, data['lbl'], train_size=0.70, random_state=42)

    x_train = x_train.ts.values
    x_train = np.array([i for i in x_train])


    y_train = pd.get_dummies(y_train).values

    x_test = x_test.ts.values
    x_test = np.array([i for i in x_test])

    y_test = pd.get_dummies(y_test).values


start_time = time.time()

# Model Hyperparameters
num_dims = x_train.shape[1]
batch_size = 64
num_classes = 2
num_epochs = 100
num_nodes = 500
activation = 'relu'
optimizer = 'Adam'

K.clear_session()
model = Sequential()
model.add(Dense(units=num_nodes, input_dim=num_dims, activation=activation))
model.add(Dropout(0.1))
model.add(Dense(num_nodes, activation=activation))
model.add(Dropout(0.2))
model.add(Dense(num_nodes, activation=activation))
model.add(Dropout(0.2))
model.add(Dense(num_nodes, activation=activation))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=200, min_lr=0.1)
print(model.summary())

history = model.fit(x_train, y_train, validation_split=0.2, batch_size=batch_size,
                    epochs=num_epochs, verbose=1,
                    callbacks=[reduce_lr])



plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(RESULTSDIR + 'accuracy_curves.png')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(RESULTSDIR + 'loss_curves.png')
plt.show()


score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))

y_pred = model.predict_classes(x_test)

tst_lbls = np.argmax(y_test, axis=1)

report = classification_report(tst_lbls, y_pred, output_dict=True)
print(report)
report = pd.DataFrame(report).transpose()
report.to_csv(RESULTSDIR + 'report.csv')

cm = confusion_matrix(tst_lbls, y_pred)

fmt = '0.2f'
ax = sns.heatmap(cm, cmap=plt.cm.Blues, annot=True, fmt=fmt)
plt.yticks(va='center')
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label')
ax.set_title('Confusion Matrix')

plt.savefig(RESULTSDIR + 'confusion_matrix.png')
plt.show()

time_elapsed = time.time() - start_time
print('Total time for training is: ', time_elapsed)
