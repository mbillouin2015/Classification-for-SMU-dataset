import plaidml.keras
plaidml.keras.install_backend()
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras import backend as K
from sklearn.metrics import classification_report

DATADIR = 'data/FordA.pickle'
# SAVEDIR = 'plots/'

data = pd.read_pickle(DATADIR)

x_train = data[data['split'] == 'TRAIN'].ts
x_train = np.array([i for i in x_train])

y_train = pd.get_dummies(data[data['split'] == 'TRAIN'].class_lbl).values

x_test = data[data['split'] == 'TEST'].ts
x_test = np.array([i for i in x_test])
y_test = pd.get_dummies(data[data['split'] == 'TEST'].class_lbl).values

# Model Hyperparameters
num_dims = x_train.shape[1]
batch_size = 16
num_classes = 2
num_epochs = 20
num_nodes = 50
activation = 'relu'
optimizer = 'Adam'

model = Sequential()
model.add(Dense(units=num_nodes, input_dim=num_dims, activation=activation))
model.add(Dropout(0.1))
model.add(Dense(num_nodes, activation=activation))
model.add(Dropout(0.2))
model.add(Dense(num_nodes, activation=activation))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

history = model.fit(x_train, y_train, validation_split=0.25, batch_size=batch_size, epochs=num_epochs, verbose=1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))