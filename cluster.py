import os
import keras
import metrics
import numpy as np
import pandas as pd
import keras.backend as K

from time import time

from keras.datasets import mnist
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
    #print(y_test)
    #y_test = keras.utils.to_categorical(y_test, 10)
    #print(y_test)

    # To stop potential randomness
    seed = 128
    rng = np.random.RandomState(seed)

    km = KMeans(n_clusters=10, n_init=20)

    km.fit(x_train)

    KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
       n_clusters=10, n_init=20, precompute_distances='auto',
       random_state=None, tol=0.0001, verbose=0)

    pred = km.predict(x_test)
    score = normalized_mutual_info_score(y_test, pred)
    print(score)
    # 0.4984422816100148