# -*- coding: utf-8 -*-

# とりあえずN個のd次元データをK通りにクラスタリングすることを考える
# mnistの場合はX = (10000, 28*28), T = (10000, 10)
# W_1 = (28*28 (= forwardlayer.inputsize (= l-1行目のUnitの数 (= z_0の列数))), z_1の列数)
## つまりW_lはz_lの列数(inputsize)を保存して、 W_l.shape = (forwardlayer.inputsize, self.inputsize)
# もちろん z_l = h^{l-]}(u_l)

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from func import * # 関数リスト


class Data(object):
    """ラベルとデータを管理するクラス
    データ X = (データ数N, データの次元d)
    ラベル T = (データ数N, 分類数K）"""
    def __init__(self, X, T):
        self.X = X
        self.T = T
        self.dim = np.shape(self.X)[1]
        self.category = np.shape(self.T)[1]
    
    def __len__(self):
        return np.shape(self.X)[0] # N


class Layer(object): # l番目のやつの情報をすべて持つだけにしようと思う
    """レイヤのクラス"""

    def __init__(self, layer_index, unit_number, activation_function=sigmoid, weight_init=0.01, learning_rate=None):
        """
        unit_number: l層目のユニット数
                      W.shape = (backward_layer.unit_number, self.unit_number)
        """
        self.unit_number = unit_number 

        # つかうやつを列挙だけしておく
        self.z = None 
        self.dW = None
        self.db = None
        self.delta = None 
        self.W = None
        self.u = None 
        self.o = None

        self.layer_index = layer_index
        self.activation_function = activation_function
        self.weight_init = weight_init
        self.learning_rate = learning_rate

    def set_relation(self, backward_layer, forward_layer):
        """input_layerとoutput_layerを教える
            forward_layer: 次のレイヤのインスタンス   l+1
            backward_layer: 前のレイヤのインスタンス  l-1
        """

        if backward_layer != None:
            self.backward_layer = backward_layer 
        if forward_layer != None:
            self.forward_layer = forward_layer
            
    def init_weight(self):
        """標準正規分布N(0,1) * e = N(0, e^2)に従うようにする
            バイアスは定数で初期化してもいいかも（分類4参照）"""
        #input_layer以外にW,bを追加

        if self.layer_index == 0:
            pass
        else:
            self.W = self.weight_init * np.random.randn(self.backward_layer.unit_number, self.unit_number)
            self.b = self.weight_init * np.random.randn(self.unit_number)
            #self.W = self.weight_init**2 * np.ones((self.backward_layer.unit_number, self.unit_number))
            #self.b = self.weight_init * np.zeros(self.unit_number)

    def forward_propagation(self):
        """forward_layerに受け渡す情報をつくる
        のはやめて普通にやる"""
        self.u = np.dot(self.backward_layer.z, self.W) + self.b
        self.z = self.activation_function(self.u)

    def back_propagation(self):
        """forward_layerの情報からdW, dbをつくる
        これneuralnet側で実装したほうがいいかも
        結局output_layerはneuralnet側で別に実装(0104)"""
        self.delta = self.activation_function(self.u, differential=True) * self.forward_layer.o
        self.o = np.dot(self.delta, self.W.T)
        self.dW = np.dot(self.backward_layer.z.T, self.delta)
        self.db = np.dot(np.ones(np.shape(self.delta)[0]), self.delta)

    def update_weight(self):
        """重みを更新"""
        self.W = self.W - self.learning_rate * self.dW
        self.b = self.b - self.learning_rate * self.db


class NeuralNet(object):
    """ニューラルネットを生成するクラス"""

    def __init__(self, input_dim, output_dim, hidden_layer_list, iteration=10, activation_function=None, loss_function=cross_entropy_error, learning_rate=0.01):
        """activation_functionがNoneなら個別に指定される必要あり
        MNISTなら input_dim = 28*28, output_dim = 10 
        データ数はどう表現しようか -> とりあえずself.data_number -> 廃止
        hidden_layer_list = [100, 24, 24]みたいな？
        """

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.loss_function = loss_function
        self.activation_function = activation_function # 暫く使わない
        self.learning_rate = learning_rate
        self.network = [] # ここにappendしていく
        self.iteration = iteration
        self.predicted_raw_data = None
        
        # hidden_layer_list = [1, 2, ..., L-1番目のレイヤ次元数]みたいに決める
        # [input_dim, `hidden_layer_list, output_dim]の順番で流れる

        self.layer_number = len(hidden_layer_list) + 2 # == self.networkの長さ 

        ###ネットワークを初期化する###

        # 1. input_layerをつくる
        self.network.append(Layer(0, self.input_dim, learning_rate=learning_rate))

        # 2. hidden_layerをつくる (とりあえず活性化関数はsigmoidのまま)
        layer_index = 1
        for layer_number in hidden_layer_list:
            self.network.append(Layer(layer_index, layer_number, learning_rate=learning_rate))
            layer_index += 1

        # 3. output_layerをつくる (ソフトマックス回帰をするのでsoftmax)
        self.network.append(Layer(self.layer_number - 1, self.output_dim, activation_function=softmax, learning_rate=learning_rate))

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

    def set_data(self, train_data, test_data):
        """Data型で渡す"""
        self.train_data = train_data
        self.test_data = test_data

    def _train(self):
        """学習を1反復行う（バッチ）"""
        N = len(self.train_data)
        X = self.train_data.X
        T = self.train_data.T

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

    def train(self):
        """iteration回_trainを実行（変数がダブっている）
            もう少し情報をprintする"""
        for i in range(self.iteration):
            self._train()
            print("iteration {} finished".format(i+1))

    def predict(self):
        """学習済みのWとbを用いてtest_dataを予測する"""
        X = self.test_data.X
        # print(np.shape(X)) -> (10000, 784)
        self.network[0].z = X

        # テストデータに対して順伝播する
        for i in range(1, self.layer_number):
            self.network[i].forward_propagation()

        self.predicted_raw_data = self.network[-1].z

        # print(np.shape(self.predicted_raw_data)) -> (10000, 10)
        print(self.predicted_raw_data[50])

    def accuracy(self):
        if self.predicted_raw_data == None:
            self.predict()
        
        # 各々のデータに対し最大値だったやつを返す
        predicted_index_list = list(np.argmax(self.predicted_raw_data, axis=1))
        index_list = list(np.argmax(self.test_data.T, axis=1))

        # print(len(predicted_index_list)) -> 10000
        # print(len(index_list)) -> 10000

        count = 0
        for i in range(len(predicted_index_list)):
            if predicted_index_list[i] == index_list[i]:
                count+=1
        
        return float(count / len(predicted_index_list))



    





    


        



        


