import numpy as np
import matplotlib.pyplot as plt
from nn import *
from mnist import *

train_data, test_data = make_train_and_test_data(60000, 10000)

net = NeuralNet(28*28, 10, [100], iteration=5, learning_rate=0.02)

net.set_data(train_data, test_data)

net.train()
net.predict()

ans = net.accuracy()

print(ans)




