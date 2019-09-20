import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

df = pd.read_csv('./creditcard.csv', sep=',')

X = df[df.columns[~df.columns.isin(['Time', 'Class'])]]
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, 
                                        y, 
                                        test_size=0.2)

print("X: ", X.shape)
print("y: ", y.shape)

from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(29, activation='relu', 
                        input_shape=(29,)))
network.add(layers.Dense(2, activation='softmax'))

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

from keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

network.fit(X_train, y_train, epochs=5, batch_size=128)
y_pred = network.predict(X_test)

matrix = confusion_matrix(
    y_test.argmax(axis=1), 
    y_pred.argmax(axis=1))
print(matrix)
