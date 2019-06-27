import matplotlib.pyplot as plt
import numpy as np
import pickle
import matplotlib.pyplot as plt

_PICKLE_FILE_1 = 'mnist_train.pickle'
_PICKLE_FILE_2 = 'mnist_test.pickle'
_PICKLE_FILE_3 = 'mnist_test_10.pickle'

class NN:
    def __init__(self, layers):
        self.m = layers
        self.l = [None for x in range(len(self.m))]
        self.w = [None for x in range(len(self.m)-1)]
        self.fails = []
        f = open(_PICKLE_FILE_1, 'rb')
        self.traindata = pickle.load(f)
        f.close()
        f = open(_PICKLE_FILE_2, 'rb')
        self.testdata = pickle.load(f)
        f.close()
        for i in range(len(self.m)-1):
            self.w[i] = np.random.randn(self.m[i+1], self.m[i]) / np.sqrt(self.m[i]/2)
    def predict(self, x):
        label = int(x[0])
        self.l[0] = np.array(self.normalize(x[1:]),ndmin=2).T
        j = 0
        for i in range(len(self.m)-2):
            j = i
            self.l[j+1] = self.h(np.dot(self.w[j], self.l[j]))
        j+=1
        self.l[j+1] = self.softmax(np.dot(self.w[j], self.l[j]))
        return self.l[j+1]
    def predict2(self, x): #(w/o softmax)
        label = int(x[0])
        self.l[0] = np.array(self.normalize(x[1:]),ndmin=2).T
        j = 0
        for i in range(len(self.m)-2):
            j = i
            self.l[j+1] = self.h(np.dot(self.w[j], self.l[j]))
        j+=1
        self.l[j+1] = np.dot(self.w[j], self.l[j])
        return self.l[j+1]
    def train(self, traindata=[], lr=0.01, epoch=1):
        dw = [None for x in range(len(self.m)-1)]
        t = [None for x in range(len(self.m)-1)]
        if not traindata:traindata = self.traindata
        for e in range(epoch):
            print("* Epoch %d" % e)
            for k, x in enumerate(traindata):
                target = np.array(np.zeros(self.m[-1]), ndmin=2).T
                target[int(x[0]),0] = 1
                self.l[0] = np.array(self.normalize(x[1:]),ndmin=2).T
                # Forward prop.
                j = 0
                for i in range(len(self.m)-2):
                    j = i
                    self.l[j+1] = self.h(np.dot(self.w[j], self.l[j]))
                j+=1
                self.l[j+1] = self.softmax(np.dot(self.w[j], self.l[j]))
                # Backprop.
                t[2] = self.l[3] - target
                dw[2] = np.dot( t[2], self.l[2].T )
                t[1] = self.l[2] * (1 - self.l[2])
                dw[1] = np.dot( np.dot(self.w[2].T, t[2]) * t[1], self.l[1].T )
                t[0] = self.l[1] * (1 - self.l[1])
                dw[0] = np.dot( np.dot( np.dot(self.w[2].T, t[2]).T, np.dot( np.dot(t[1], np.array(np.full(self.m[2],1),ndmin=2)), self.w[1])).T * (t[0] * ( 1 - t[0])) , self.l[0].T )

                #dw[1] = np.dot( (self.l[2] - target), self.l[1].T )
                #dw[0] = np.dot( np.dot(self.w[1].T, (self.l[2] - target)) * self.l[1] * (1 - self.l[1]), self.l[0].T )
                for i in range(len(self.m)-1):
                    self.w[i] -= lr * dw[i]
                if (k+1) % 30000 == 0:self.print_acc()
    def print_acc(self):
        self.fails = [[] for x in range(10)]
        matches = 0
        for x in self.testdata:
            label = int(x[0])
            result = self.predict(x)
            prediction = np.argmax(result)
            #print("[" + str(label) + " / " + str(prediction) + " / " + str(result[prediction]))
            if label == prediction:
                matches += 1
            else:
                self.fails[label].append(np.array([prediction, *x[1:]], dtype='uint8'))
        print("Acc : " + str(matches/len(self.testdata)*100) + "%")
    def h(self, x):return 1.0/(1.0+np.exp(-x))
    def normalize(self, x):return (x / 255.0) * 0.99 + 0.01
    def softmax(self, x):return np.exp(x) / np.sum(np.exp(x))
    def graph(self, n=10):
        if not self.fails:self.print_acc()
        fig, axes = plt.subplots(10, 10)
        fig.set_size_inches(9.6,7.2)
        for i in range(n*10):
            n,m=i%10,i//10
            axes[n,m].set_yticklabels([])
            axes[n,m].set_xticklabels([])
            if n>len(self.fails[m])-1:continue
            axes[n,m].imshow(self.fails[m][n][1:].reshape((28,28)), cmap='gray')
            axes[n,m].set_title('%d' % (self.fails[m][n][0]), size='x-small', pad=1)
        plt.tight_layout()
        plt.show(block=False)


#a=NN((784,100,10))
a=NN((784,100,50,10))

f=open('w_3', 'rb')
a.w=pickle.load(f)
