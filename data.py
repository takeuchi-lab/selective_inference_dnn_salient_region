from abc import abstractproperty
from typing import Tuple
import numpy as np
import tensorflow as tf
from keras.utils.np_utils import to_categorical


def generate(n, d, mu_1, mu_2):
    list_X_train = []
    list_X_label = []

    for _ in range(n):
        X_train = []
        X_label = []

        for i in range(d):
            if (i < d / 4) or (i >= 3 * d / 4):
                vec_train = []
                vec_label = []
                gen_vec = list(np.random.normal(mu_1, 1, d))

                for j in range(d):
                    vec_train.append([gen_vec[j]])
                    vec_label.append([False])

                X_train.append(list(vec_train))
                X_label.append(list(vec_label))
            else:
                vec_train = []
                vec_label = []

                for j in range(d):
                    if (j < d / 4) or (j >= 3 * d / 4):
                        vec_train.append([float(np.random.normal(mu_1, 1))])
                        vec_label.append([False])
                    else:
                        vec_train.append([float(np.random.normal(mu_2, 1))])
                        vec_label.append([True])

                X_train.append(list(vec_train))
                X_label.append(list(vec_label))

        list_X_train.append(np.array(X_train))
        list_X_label.append(np.array(X_label))

    return np.array(list_X_train), np.array(list_X_label)


def generate_data(shape, n: int, signal_1, signal_2) -> tuple:
    """generate a dataset for fcnn

    Args:
        shape (Tuple): the image shape of the input
        n (int): the number of samples to generate
    """
    X_train = []
    Y_train = []
    for i in range(n):
        a = shape[0] // 4

        X = np.random.normal(signal_1, 1, shape)
        Y = np.zeros(shape)
        abnormal_x = np.random.randint(0, shape[0] - a)
        abnormal_y = np.random.randint(0, shape[1] - a)
        X[
            abnormal_x: abnormal_x + a, abnormal_y: abnormal_y + a, 0
        ] = np.random.normal(signal_2, 1, (a, a))
        Y[abnormal_x: abnormal_x + a,
            abnormal_y: abnormal_y + a, 0] = np.ones((a, a))

        X_train.append(X)
        Y_train.append(Y)

    return np.array(X_train), np.array(Y_train)


def generate_data_classification(shape, n: int, signal_1, signal_2, c=None) -> tuple:
    """generate a dataset for fcnn

    Args:
        shape (Tuple): the image shape of the input
        n (int): the number of samples to generate
    """

    X_train = []
    Y_train = []

    if c is not None:
        if c:
            X = np.random.normal(signal_1, 1, shape)
            return np.array([X])
        else:
            a = shape[0] // 2
            X = np.random.normal(signal_1, 1, shape)
            X_true = np.zeros(shape)
            abnormal_x = np.random.randint(0, shape[0] - a+1)
            abnormal_y = np.random.randint(0, shape[1] - a+1)
            X[
                abnormal_x: abnormal_x + a, abnormal_y: abnormal_y + a, 0
            ] = np.random.normal(signal_2, 1, (a, a))
            X_true[
                abnormal_x: abnormal_x + a, abnormal_y: abnormal_y + a, 0
            ] = 1

            return np.array([X]),np.array([X_true])

            # generate negative dataset
    for i in range(n//2):
        X = np.random.normal(signal_1, 1, shape)
        X_train.append(X)
        Y_train.append(0)

    # generate positive dataset
    for i in range(n//2):
        a = shape[0] // 4

        X = np.random.normal(signal_1, 1, shape)
        Y = np.zeros(shape)
        abnormal_x = np.random.randint(0, shape[0] - a)
        abnormal_y = np.random.randint(0, shape[1] - a)
        X[
            abnormal_x: abnormal_x + a, abnormal_y: abnormal_y + a, 0
        ] = np.random.normal(signal_2, 1, (a, a))

        X_train.append(X)
        Y_train.append(1)

    # 配列をシャッフル
    index = np.arange(0, n)
    rng = np.random.default_rng(0)
    rng.shuffle(index)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    X_train = X_train[index, :, :, :]
    Y_train = Y_train[index]

    return np.array(X_train), np.array(Y_train)


def generate_data_multi(shape, n: int, signal_0, signal_1, signal_2) -> tuple:
    """generate a dataset for fcnn

    Args:
        shape (Tuple): the image shape of the input
        n (int): the number of samples to generate
    """
    X_train = []
    Y_train = []
    for i in range(n):
        a = shape[0] // 4

        X = np.random.normal(signal_0, 1, shape)
        Y = np.zeros(shape)
        abnormal_x = np.random.randint(0, shape[0] - a)
        abnormal_y = np.random.randint(0, shape[1] - a)
        X[
            abnormal_x: abnormal_x + a, abnormal_y: abnormal_y + a, 0
        ] = np.random.normal(signal_1, 1, (a, a))
        Y[abnormal_x: abnormal_x + a,
            abnormal_y: abnormal_y + a, 0] = np.ones((a, a))

        a = shape[0] // 4
        abnormal_x = np.random.randint(0, shape[0] - a)
        abnormal_y = np.random.randint(0, shape[1] - a)
        X[
            abnormal_x: abnormal_x + a, abnormal_y: abnormal_y + a, 0
        ] = np.random.normal(signal_2, 1, (a, a))
        Y[abnormal_x: abnormal_x + a,
            abnormal_y: abnormal_y + a, 0] = 2*np.ones((a, a))

        X_train.append(X)
        Y_train.append(Y)

    Y_train = np.array(Y_train)
    Y = to_categorical(tf.constant(Y_train))

    return np.array(X_train), Y


def generate_data_old(shape, n: int, signal_1, signal_2) -> tuple:
    """generate a dataset for fcnn

    Args:
        shape (Tuple): the image shape of the input
        n (int): the number of samples to generate
    """
    X_train = []
    Y_train = []
    for i in range(n):

        X = np.random.normal(signal_1, 1, shape)
        Y = np.zeros(shape)
        abnormal_x = np.random.randint(0, shape[0] - 1)
        abnormal_y = np.random.randint(0, shape[1] - 1)
        X[
            abnormal_x: abnormal_x + 2, abnormal_y: abnormal_y + 2, 0
        ] = np.random.normal(signal_2, 1, (2, 2))
        Y[abnormal_x: abnormal_x + 2,
            abnormal_y: abnormal_y + 2, 0] = np.ones((2, 2))

        X_train.append(X)
        Y_train.append(Y)

    return np.array(X_train), np.array(Y_train)
