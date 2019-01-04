# -*- coding: utf-8 -*-

# とりあえずN個のd次元データをK通りにクラスタリングすることを考える
# mnistの場合はX = (10000, 28*28), T = (10000, 10)
# W_1 = (28*28 (= forwardlayer.inputsize (= l-1行目のUnitの数 (= z_0の列数))), z_1の列数)
## つまりW_lはz_lの列数(inputsize)を保存して、 W_l.shape = (forwardlayer.inputsize, self.inputsize)
# もちろん z_l = h^{l-]}(u_l)


import numpy as np
import matplotlib.pyplot as plt
from func import * # 関数リスト

class Layer(object): # l番目のやつの情報をすべて持つだけにしようと思う
    """レイヤの親クラス"""

    def __init__(self, layer_index, unit_number, activation_function=sigmoid, weight_init=0.01, learning_rate=0.01):
        """
        forward_layer: 次のレイヤのインスタンス
        input_size: int いわゆる i のこと
        output_size:int いわゆる j のこと
        activation_functionがNoneなら個別に活性化関数指定する
        """
        self.input_size = unit_number

        # つかうやつを列挙だけしておく
        self.z = None 
        self.dW = None #dJ/dW^l
        self.db = None # dJ/db^l
        self.delta = None 
        self.W = None
        self.u = None 
        self.o = None
        self.N = np.shape(self.z)[0] #ぜったいにここじゃないどこか

        self.layer_index = layer_index
        self.activation_function = activation_function
        self.weight_init = weight_init
        self.learning_rate = learning_rate

    def set_relation(self, backward_layer, forward_layer):
        """input_layerとoutput_layerの処理を少し変える"""
        if backward_layer != None:
            self.backward_layer = backward_layer # l-1
            self.output_size = self.backward_layer.input_size
        if forward_layer != None:    
            self.forward_layer = forward_layer   # l+1
            
    def init_weight(self):
        """標準正規分布N(0,1) * e = N(0, e^2)に従うようにする
            バイアスは定数で初期化してもいいかも（分類4参照）"""
        self.W = self.weight_init * np.random.randn(self.input_size, self.output_size)
        self.b = self.weight_init * np.random.randn(self.output_size)
    
    def forward_propagation(self):
        """forward_layerに受け渡す情報をつくる
        のはやめて普通にやる"""
        self.u = np.dot(self.backward_layer.z, self.W) + self.b
        self.z = self.backward_layer.activation_function(self.u)

    def back_propagation(self, output=False):
        """forward_layerの情報からdW, dbをつくる
        これnet側で実装したほうがいいかも"""
        self.delta = self.activation_function(self.u, differential=True) * self.forward_layer.o
        self.o = np.dot(self.delta, self.W.T)
        self.dW = np.dot(self.backward_layer.z.T, self.delta)
        self.db = np.dot(np.ones(self.N), self.delta)

    def update_weight(self):
        """重みを更新"""
        self.W = self.W - self.learning_rate * self.W
        self.b = self.b - self.learning_rate * self.b


class NeuralNet(object):
    """ニューラルネットを生成するクラス"""

    def __init__(self, input_shape, output_shape, layer_list, iteration, activation_function=None, loss_function=cross_entropy_error, learning_rate=0.01):
        """activation_functionがNoneなら個別に指定される必要あり
        MNISTなら input_shape = 28*28, output_shape = 10 データ数10000はどう表現しようか
        layer_list = [100, 24, 24]みたいな？
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.loss_function = loss_function
        self.activation_function = activation_function # 暫く使わない
        self.learning_rate = learning_rate
        self.network = [] # ここにappendとかしていく
        self.iteration = iteration
        
        # layer_list = [1, 2, ..., L-1番目のレイヤ次元数]みたいに決める
        # [input_shape, `layer_list, output_shape]の順番で流れる
        self.layer_number = len(layer_list) + 2 # == self.networkの長さ

        ###ネットワークを初期化する###

        # 1. input_layerをつくる
        self.network.append(Layer(0, self.input_shape))

        # 2. hidden_layerをつくる (とりあえず活性化関数はsigmoid)
        layer_index = 1
        for layer_number in layer_list:
            self.network.append(Layer(layer_index, layer_number))
            layer_index += 1

        # 3. output_layerをつくる
        self.network.append(Layer(self.layer_number - 1, self.output_shape))

        ## 現時点で self.network = [input_layer, layer1, ..., layer5, output_layer]みたいになってる

        # 4. layer間の親子関係を設定
        for i in range(self.layer_number):
            if i==0:
                self.network[i].set_relation(None, self.network[i+1])
            elif i==self.layer_number-1:
                self.network[i].set_relation(self.network[i-1], None)
            else:
                self.network[i].set_relation(self.network[i-1], self.network[i+1])
        
        # 5. 重みの初期化
        for layer in self.network:
            layer.init_weight()

    def train(self, X, T):
        """学習を1反復行う（バッチ）"""
        N = np.shape(X)[0]

        # 1. input_layerのzを初期化
        self.network[0].z = X

        # 2. forward propagation
        for i in range(1, self.layer_number):
            self.network[i].forward_propagation()

        # 3. output_layerにおける誤差を計算
        self.network[-1].delta = (self.network[-1].z - T) / N
        self.network[-1].dW = np.dot(self.network[-2].z.T, self.network[-1].delta)
        self.network[-1].db = np.dot(np.ones(N), self.network[-1].delta)
        self.network[-1].o = np.dot(self.network[-1].delta, self.network[-1].W.T)

        # 4. back propagation
        for i in range(self.layer_number-2, 0, -1): # hidden_layerのindexを逆順
            self.network[i].back_propagation()
        
        # 5 重みの更新
        for i in range(1, self.layer_number):
            self.network[i].update_weight()




        



        


