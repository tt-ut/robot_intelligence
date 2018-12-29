#coding: utf-8

import numpy as np
import matplotlib.pyplot as plt 
from sklearn.datasets import fetch_mldata

# mnistの手書き数字データをダウンロード

def make_mnist_dataset():
    mnist = fetch_mldata('MNIST original', data_home="mnist/")
    return mnist






