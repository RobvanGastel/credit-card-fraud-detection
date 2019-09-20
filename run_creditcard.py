import numpy as np
import pandas as pd

df = pd.read_csv('./creditcard.csv', sep=',')

X = df[df.columns[~df.columns.isin(['Time', 'Class'])]]
y = df['Class']

print(X.shape)
print(y.shape)

from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(29, activation='relu', input_shape=(29,)))
network.add(layers.Dense(2, activation='softmax'))

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

from keras.utils import to_categorical

y_bin = to_categorical(y)

network.fit(X, y_bin, epochs=5, batch_size=128)