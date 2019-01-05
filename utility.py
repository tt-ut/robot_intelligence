import pickle

# 学習中のNeuralNetの保存、読み込みをする

def save_net(name, net):
    with open(name, 'wb') as f:
        pickle.dump(net , f)

def load_net(name):
    with open(name, 'rb') as f:
        net = pickle.load(f)
        return net


