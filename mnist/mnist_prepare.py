import pickle
import numpy as np

with open('mnist_train.csv','r') as f: 
    rawdata = f.readlines()
traindata = [np.array(x.strip().split(','), dtype=np.uint8) for x in rawdata]
with open('mnist_train.pickle','wb') as f:
    pickle.dump(traindata, f)

with open('mnist_test.csv','r') as f:
    rawdata = f.readlines()
testdata = [np.array(x.strip().split(','), dtype=np.uint8) for x in rawdata]
with open('mnist_test.pickle','wb') as f:
    pickle.dump(testdata, f)
