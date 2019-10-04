import os
import keras
import metrics
import numpy as np
import pandas as pd
import keras.backend as K

from time import time

from keras.datasets import mnist
from keras import callbacks
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Dense, Input
from keras.initializers import VarianceScaling
from keras.engine.topology import Layer, InputSpec

from scipy.misc import imread
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, normalized_mutual_info_score

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1,784).astype('float32')
    x_train /= 255
    x_test = x_test.reshape(-1,784).astype('float32')
    x_test /= 255
    #y_train = keras.utils.to_categorical(y_train, 10)
    #y_test = keras.utils.to_categorical(y_test, 10)

    # To stop potential randomness
    seed = 128
    rng = np.random.RandomState(seed)

    # this is our input placeholder
    input_img = Input(shape=(784,))

    # "encoded" is the encoded representation of the input
    encoded = Dense(500, activation='relu')(input_img)
    encoded = Dense(500, activation='relu')(encoded)
    encoded = Dense(2000, activation='relu')(encoded)
    encoded = Dense(10, activation='sigmoid')(encoded)

    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(2000, activation='relu')(encoded)
    decoded = Dense(500, activation='relu')(decoded)
    decoded = Dense(500, activation='relu')(decoded)
    decoded = Dense(784)(decoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)

    autoencoder.summary()

    #  this model maps an input to its encoded representation
    encoder = Model(input_img, encoded)

    autoencoder.compile(optimizer='adam', loss='mse')

    train_history = autoencoder.fit(x_train, x_train, epochs=100, batch_size=2048, validation_data=(x_test, x_test)) #epoch=500

    pred_auto_train = encoder.predict(x_train)
    pred_auto = encoder.predict(x_test)

    km = KMeans(n_clusters=10, n_init=20)
    km.fit(pred_auto_train)
    pred = km.predict(pred_auto)

    score = normalized_mutual_info_score(y_test, pred)
    print(score)
    # 0.7055804338307965