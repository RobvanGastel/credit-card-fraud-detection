import numpy as np
import pandas as pd

df = pd.read_csv('../creditcard.csv', sep=',')

from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(28, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(2, activation='softmax'))

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

