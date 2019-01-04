# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from func import * # 関数リスト

class Layer(object): # l番目のやつの情報をすべて持つだけにしようと思う
    """レイヤの親クラス"""

    def __init__(self, input_size, forward_layer, layer_number, activation_function=None, weight_init=0.01, learning_rate=0.01):
        """
        forward_layer: 次のレイヤのインスタンス
        input_size: int いわゆる i のこと
        output_size:int いわゆる j のこと
        activation_functionがNoneなら個別に活性化関数指定する
        """
        self.input_size = input_size
        self.forward_layer = forward_layer
        self.output_size = self.forward_layer.input_size
        self.layer_number = layer_number
        self.activation_function = activation_function
        self.weight_init = weight_init
        self.learning_rate = learning_rate

    def init_weight(self):
        """標準正規分布N(0,1) * e = N(0, e^2)に従うようにする
            バイアスは定数で初期化してもいいかも（分類4参照）"""
        self.W = self.weight_init * np.random.randn(self.input_size, self.output_size)
        self.b = self.weight_init * np.random.randn(self.output_size)
    
    def forward_propagation(self, _z):
        """forward_layerに受け渡す情報をつくる"""
        z = _z.copy()
        forward_u = np.dot(z, self.W) + self.b
        forward_z = self.activation_function(forward_u)
        return forward_z
    
    def back_propagation(self):
        """forward_layerの情報からdW, dbをつくる"""
        
    def update(self):
        """重みを更新"""
        self.W = self.W - self.learning_rate * self.W
        self.b = self.b - self.learning_rate * self.b

class InputLayer(Layer):
    def __init__(self, forward_layer, activation_function, weight_init, learning_rate):
        super().__init__(input_size, forward_layer, activation_function, weight_init, learning_rate)
        self.layer_number = 0


class NeuralNet(object):
    """ニューラルネットを生成するクラス"""

    def __init__(self, input_shape, output_shape, layer_list, activation_function=sigmoid, loss_function=cross_entropy_error, learning_rate=0.01):
        """activation_functionがNoneなら個別に指定される必要あり"""
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.loss_function = loss_function
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.network = [] # ここにappendとかしていく

        
