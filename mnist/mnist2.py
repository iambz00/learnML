import numpy as np

data_file = open("mnist_train.csv", "r")
training_data = data_file.readlines()
data_file.close()

test_data_file = open("mnist_test.csv", "r")
test_data = test_data_file.readlines()
test_data_file.close()

class DeepNeuralNetwork:
    def __init__(self, input_layers, hidden_layers, output_layers):
        self.inputs = input_layers
        self.hiddens = hidden_layers
        self.outputs = output_layers
        self.test_data = None

        self.wih = np.random.randn(self.inputs, self.hiddens) / np.sqrt(self.inputs/2)
        self.who = np.random.randn(self.hiddens, self.outputs) / np.sqrt(self.hiddens/2)

# feed-forward를 진행한다.
    def predict(self, x):
        # 문자열을 float array로 바꾸는 과정
        data = self.normalize(np.asfarray(x.split(',')))

        # 0번은 라벨이기 때문에 날렸다.
        data = data[1:]

        layer_1 = self.sigmoid(np.dot(data, self.wih))
        output = self.sigmoid(np.dot(layer_1, self.who))
        return output

    # training_data로 학습을 진행한다.
    def train(self, training_data, lr=0.01, epoch=1):
        for ech in range(0, epoch):
            for i, x in enumerate(training_data):
                target = np.array(np.zeros(self.outputs) + 0.01, ndmin=2)
                target[0][int(x[0])] = 0.99
                x = self.normalize(np.asfarray(x.split(",")))

                # feed forward
                l1 = self.sigmoid(np.dot(x[1:], self.wih))
                l2 = self.sigmoid(np.dot(l1, self.who))

                # back propagation alogrithm.
                l2_e = (target - l2) * (l2 * (1 - l2))
                l1_e = l2_e.dot(self.who.T) * (l1 * (1 - l1))

                # update
                self.who = self.who + lr * l2_e.T.dot(np.array(l1, ndmin=2)).T
                self.wih = self.wih + lr * l1_e.T.dot(np.array(x[1:], ndmin=2)).T

                if i % 2000 == 0 :
                    self.print_accuracy()

    # 현재 신경망의 정확도를 출력한다.
    def print_accuracy(self):
        matched = 0

        for x in self.test_data:
            label = int(x[0])
            predicted = np.argmax(self.predict(x))
            if label == predicted :
                matched = matched + 1
        print('현재 신경망의 정확도 : {0}%'.format(matched/len(self.test_data)*100))

    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))

    def normalize(self, x):
        return (x / 255.0) * 0.99 + 0.01

network = DeepNeuralNetwork(input_layers=784, hidden_layers=100, output_layers=10)
network.test_data = test_data
network.train(training_data, lr=0.01, epoch=1)
