#coding: utf-8

import numpy as np
import matplotlib.pyplot as plt 
from sklearn.datasets import fetch_mldata
import copy
from nn import Data

# mnistの手書き数字データをダウンロード

def make_onehot_for_mnist(target):
    """mnistのtargetをonehotにする"""
    CATEGORY_NUMBER = 10
    T = np.zeros((len(target), CATEGORY_NUMBER))
    for i in range(len(target)):
        T[i][int(target[i])] = 1
    return T

def make_mnist_data():
    """mnistのデータをすべて都合いい形にする"""
    mnist = fetch_mldata('MNIST original', data_home="mnist/")
    data = copy.deepcopy(mnist.data / 255.0) # (70000, 784)
    label = make_onehot_for_mnist(mnist.target) # (70000, 10)

    return data, label

def make_train_and_test_data(N, M):
    """ランダムにN+M個のデータを取り出して、訓練N個、テストM個"""
    full_data, full_label = make_mnist_data()
    full_data_number = np.shape(full_data)[0]

    if full_data_number < N+M:
        print("割り振れるデータ数は{}個まで".format(full_data_number))

    mask = np.random.choice(list(range(full_data_number)), N+M, replace=False)
    # 実験のためにランダム部分をなくす
    # mask = list(range(M+N))
    train_mask = mask[:N]
    test_mask = mask[N:]

    # Data型に格納する
    train_dataset = Data(full_data[train_mask], full_label[train_mask])
    test_dataset = Data(full_data[test_mask], full_label[test_mask])
    return train_dataset, test_dataset


def show_image_for_mnist(data, label=None):
    plt.axis('off')
    if label==None:
        plt.imshow(np.reshape(data, (28, 28)))
    else:
        plt.title('%i' % label)
    plt.show()









