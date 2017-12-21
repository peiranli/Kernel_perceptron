import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import comb
from sklearn.linear_model import Perceptron
from mlxtend.plotting import plot_decision_regions

data1 = np.loadtxt("data1.txt",dtype=np.int)
data2 = np.loadtxt("data2.txt",dtype=np.int)
x1 = data1[:, 0:2]
y1 = data1[:, 2]
x2 = data2[:, 0:2]
y2 = data2[:, 2]

def get_label(alpha, b, X, x, y):
    sum = 0
    for i in range(len(alpha)):
        sum += alpha[i]*y[i]*math.pow((1+np.dot(x[i],X)), 2)
    linear_function = sum + b
    if(linear_function > 0):
        return 1
    else:
        return -1

def quadratic_kernel(X, Y):
    w = [0]*5
    b = 0
    done = 0
    while(done == 0):
        updates = 0
        for i in range(len(X)):
            linear_function = np.dot(w, phi(X[i])) + b
            if(Y[i]*linear_function <= 0):
                w += Y[i]*phi(X[i])
                b += Y[i]
                updates += 1
        if(updates == 0):
            done = 1
    return (w, b)

def compute_RBF(x1, x2, sigma):
    x = np.subtract(x1,x2)
    k = np.dot(x, x)/ ( 2*sigma**2 )
    return np.exp(-k)

def rbf_kernel(X, Y, sigma):
    alpha = [0]*len(Y)
    b = 0
    done = 0
    while(done == 0):
        print("training...")
        updates = 0
        for i in range(len(X)):
            sum = 0
            for j in range(len(alpha)):
                sum += alpha[j]*Y[j]*compute_RBF(X[j],X[i],sigma)
            linear_function=sum + b
            if(Y[i]*linear_function <= 0):
                alpha[i] += 1
                b += Y[i]
                updates += 1
        if(updates == 0):
            done = 1
    return (alpha, b)

def phi(x):
    phi = [None]*5
    phi[0] = x[0]
    phi[1] = x[1]
    phi[2] = math.pow(x[0],2)
    phi[3] = math.pow(x[1],2)
    phi[4] = x[0]*x[1]
    return np.array(phi)

def L2_distance(x1,x2):
    distance = 0
    for index in range(len(x1)):
        distance_i = math.pow((x1[index]-x2[index]),2)
        distance += distance_i
    return distance

def graph(formula, x_range):
    x = np.array(x_range)
    y = formula(x)  # <- note now we're calling the function 'formula' with x
    plt.plot(x, y)

def my_formula(x, alpha, b):
    sum = 0
    for j in range(len(alpha)):
        sum += alpha[j]*Y[j]*math.pow(1+np.dot(X[j],x),2)
    return sum+b

def z_func(x1, x2, w, b):
    return (w[0]*x1 + w[1]*x2 + w[2]*x1**2 + w[3]*x2**2 + w[4]*x1*x2 + b)

def predict(x, alpha, b, X, Y, sigma):
    sum = [0]*2
    for j in range(len(alpha)):
        sum += alpha[j]*Y[j]*compute_RBF(X[i],x, sigma)
    linear_function=sum + b
    return linear_function

x1_1 = []
x1_2 = []
for i in range(len(y1)):
    if(y1[i] == 1):
        x1_1.append(x1[i])
    elif(y1[i] == -1):
        x1_2.append(x1[i])
x1_1 = np.array(x1_1)
x1_2 = np.array(x1_2)

x2_1 = []
x2_2 = []
for i in range(len(y2)):
    if(y2[i] == 1):
        x2_1.append(x2[i])
    elif(y2[i] == -1):
        x2_2.append(x2[i])
x2_1 = np.array(x2_1)
x2_2 = np.array(x2_2)

#temp.py
alpha1, b1 = rbf_kernel(x1, y1, 1)
alpha2, b2 = rbf_kernel(x2, y2, 1)

delta = 0.005
x1_min, x1_max = 0, 11
x2_min, x2_max = 0, 11
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, delta), np.arange(x2_min, x2_max, delta))
Z = predict(np.c_[xx1.ravel(), xx2.ravel()], alpha1, b1, x1, y1,1)
Z = Z.reshape(xx1.shape)
plt.pcolormesh(xx1, xx2, Z, cmap=plt.cm.Pastel2,vmin=0, vmax=2)
plt.scatter(x1_1[:,0],x1_1[:,1],color='red',marker='>')
plt.scatter(x1_2[:,0],x1_2[:,1],color='blue',marker='o')
plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.show()

x3_min, x3_max = 0, 11
x4_min, x4_max = 0, 11
xx3, xx4 = np.meshgrid(np.arange(x3_min, x3_max, delta), np.arange(x4_min, x4_max, delta))
Z = predict(np.c_[xx3.ravel(), xx4.ravel()], alpha2, b2, x2, y2,1)
Z = Z.reshape(xx3.shape)
plt.pcolormesh(xx3, xx4, Z, cmap=plt.cm.Pastel2,vmin=0, vmax=2)
plt.scatter(x2_1[:,0],x2_1[:,1],color='red',marker='>')
plt.scatter(x2_2[:,0],x2_2[:,1],color='blue',marker='o')
plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.show()
