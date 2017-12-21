import numpy as np
import math
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.linear_model import Perceptron

data = np.loadtxt("data0.txt",dtype=np.int)
x = data[:, 0:2]
y = data[:, 2]

def get_label(w, b, x):
    labels = [None]*4
    for i in range(len(w)):
        labels[i] = np.dot(w[i],x) + b[i]
    return np.argmax(labels)

def multi_perceptron(X, Y):
    w = np.zeros((4, len(X[0])))
    b = np.zeros(4)
    done = 0
    while(done == 0):
        updates = 0
        for i in range(len(X)):
            label = get_label(w, b, X[i])
            if(label != Y[i]):
                w[Y[i]] = w[Y[i]] + X[i]
                b[Y[i]] = b[Y[i]] + 1
                w[label] = w[label] - X[i]
                b[label] = b[label] - 1
                updates += 1
        print(updates)
        if(updates == 0):
            done = 1
    return (w, b)

w, b = multi_perceptron(x,y)

perceptron = Perceptron()
perceptron.fit(x,y)
perceptron.coef_ = w
perceptron.intercept_ = b

plot_decision_regions(x,y,clf=perceptron,legend=2)
plt.xlabel("feature one")
plt.ylabel("feature two")
plt.show()
