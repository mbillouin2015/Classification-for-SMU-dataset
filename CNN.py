tensorflow = 0

if tensorflow:
    from tensorflow.keras.layers import Dense, Conv1D, Conv2D, BatchNormalization, Activation, Flatten, GlobalAveragePooling2D
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras import backend as K
else:
    import plaidml.keras
    plaidml.keras.install_backend()
    import os
    os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

    from keras.layers import Dense, Conv1D, Conv2D, BatchNormalization, Activation, Flatten, GlobalAveragePooling2D
    from keras.models import Sequential
    from keras.optimizers import Adam
    from keras import backend as K

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import seaborn as sns

RESULTSDIR = 'results/CNN/'
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

    x_train, x_test, y_train, y_test = train_test_split(data, data['lbl'], train_size=0.70, random_state=42)

    x_train = x_train.ts.values
    x_train = np.array([i for i in x_train])

    y_train = pd.get_dummies(y_train).values

    x_test = x_test.ts.values
    x_test = np.array([i for i in x_test])

    y_test = pd.get_dummies(y_test).values


x_train = x_train.reshape(x_train.shape + (1,))
x_test = x_test.reshape(x_test.shape + (1,))

# Model Hyperparameters
activation = 'relu'
batch_size = 16
optimizer = 'Adam'
padding = 'same'
loss = 'categorical_crossentropy'
num_epochs = 20
init_mode = 'lecun_normal'

model = Sequential()

nDims = x_train.shape[1:]
model.add(Conv1D(filters=128, kernel_size=8, input_shape=nDims, padding=padding, activation=activation))
model.add(BatchNormalization())
model.add(Conv1D(filters=256, kernel_size=5, padding=padding, activation=activation))
model.add(BatchNormalization())
model.add(Conv1D(filters=128, kernel_size=3, padding=padding, activation=activation))
model.add(BatchNormalization())
model.add(Flatten())
# model.add(Dense(50, activation='relu'))
model.add(Dense(2, activation='softmax'))


# nDims = x_train.shape[1:]
# model.add(Conv2D(128, (8, 1), input_shape=nDims, padding='same', activation=activation,
#                  kernel_initializer=init_mode))
# model.add(BatchNormalization())
# model.add(Conv2D(256, (5, 1), padding='same', activation=activation, kernel_initializer=init_mode))
# model.add(BatchNormalization())
# model.add(Conv2D(256, (3, 1), padding='same', activation=activation, kernel_initializer=init_mode))
# model.add(BatchNormalization())
# model.add(GlobalAveragePooling2D())
# model.add(Dense(units=2, activation='softmax'))

model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
print(model.summary())


history = model.fit(x_train, y_train, validation_split=0.25, batch_size=batch_size, epochs=num_epochs)
# history = model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=1)


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
