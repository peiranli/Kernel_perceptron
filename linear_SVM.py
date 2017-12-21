import numpy as np
from sklearn.svm import SVC, LinearSVC
from mnist import MNIST
mndata = MNIST('samples')
trainX, trainY = mndata.load_training()
#trainX is 60000 rows, 784 columns
#trainY is the corresponding labels with 60000 rows

testX, testY = mndata.load_testing()
#testX is 10000 rows, 784 columns
#testY is the corresponding labels with 10000 rows

C = [0.01,0.1,1.0,10.0,100.0]
for i in range(len(C)):
    clf = LinearSVC(C=C[i],loss='hinge')
    clf.fit(trainX, trainY)
    z_train = clf.predict(trainX)
    err_train = 0
    for j in range(len(trainY)):
        if(z_train[j] != trainY[j]):
            err_train += 1
    err_train_rate = float(err_train)/len(trainY)
    print(C[i])
    print(err_train_rate)
    z_test = clf.predict(testX)
    err_test = 0
    for j in range(len(testY)):
        if(z_test[j] != testY[j]):
            err_test += 1
    err_test_rate = float(err_test)/len(testY)
    print(C[i])
    print(err_test_rate)
