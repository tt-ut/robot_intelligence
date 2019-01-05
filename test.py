import numpy as np
import matplotlib.pyplot as plt
from nn import *
from mnist import *

train_data, test_data = make_train_and_test_data(60000, 10000)
print("aa")
net = NeuralNet(28*28, 10, [10], 60000, iteration=20)

net.set_data(train_data, test_data)
print("ここまで動く")

net.train_loop()
ans = net.accuracy()

print(ans)




