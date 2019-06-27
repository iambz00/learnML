import numpy as np
import pickle
import matplotlib.pyplot as plt

_DATA_TRAIN = 'mnist_train.pickle'
_DATA_TEST = 'mnist_test.pickle'

class NN:
    def __init__(self):
        self.w = [None] * 2
        self.w[0] = np.random.randn(100, 784) / np.sqrt(784/2)
        self.w[1] = np.random.randn(10, 100) / np.sqrt(100/2)
        with open(_DATA_TRAIN, 'rb') as f: self.traindata = pickle.load(f)
        with open(_DATA_TEST, 'rb') as f: self.testdata = pickle.load(f)
    def guess(self, x):
        w = self.w
        l = [None] * 3
        l[0] = np.array(self.normalize(x[1:]),ndmin=2).T
        l[1] = self.h(np.dot(w[0], l[0]))
        l[2] = self.softmax(np.dot(w[1], l[1]))
        return l[2]
    def rate(self, testdata=[]):
        if not testdata: testdata = self.testdata
        matches = 0
        for x in self.testdata:
            label = x[0]
            result = np.argmax(self.guess(x))
            if label == result: matches += 1
        print(" Accuracy: %.2f%%" % (matches/len(self.testdata)*100) )
    def train(self, traindata=[], testdata=[], lr=0.01, epoch=1, check=5000):
        w = self.w
        dw = [None] * 2
        l = [None] * 3
        if not traindata: traindata = self.traindata
        for e in range(epoch):
            print("* Epoch %d" % (e+1))
            for k, x in enumerate(traindata):
                t = np.array(np.zeros(10), ndmin=2).T
                label = x[0]
                t[label, 0] = 1
                # Forward
                l[0] = np.array(self.normalize(x[1:]),ndmin=2).T
                l[1] = self.h(np.dot(w[0], l[0]))
                l[2] = self.softmax(np.dot(w[1], l[1]))
                # Backward
                dw[1] = np.dot( (l[2] - t) * t, l[1].T )
                dw[0] = np.dot( np.dot(w[1].T, (l[2] - t)) * l[1] * (1 - l[1]), l[0].T )
                w[1] -= lr * dw[1]
                w[0] -= lr * dw[0]
                if k % check == 0: self.rate(testdata)
    def h(self, x):return 1.0/(1.0+np.exp(-x))
    def normalize(self, x):return (x / 255.0)
    def softmax(self, x):return np.exp(x) / np.sum(np.exp(x))

a=NN()
a.train()
