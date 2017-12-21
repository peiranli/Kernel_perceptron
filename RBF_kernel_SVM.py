import numpy as np
import math
from sklearn.svm import SVC, LinearSVC
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.linear_model import Perceptron

data1 = np.loadtxt("data1.txt",dtype=np.int)
data2 = np.loadtxt("data2.txt",dtype=np.int)
x1 = data1[:, 0:2]
y1 = data1[:, 2]
x2 = data2[:, 0:2]
y2 = data2[:, 2]

# Create a SVC classifier using an RBF kernel
svm = SVC(kernel='rbf', random_state=0, gamma=.01, C=1)
# Train the classifier
svm.fit(x1, y1)

# Visualize the decision boundaries
plot_decision_regions(x1, y1, clf=svm)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

svm.fit(x2, y2)

plot_decision_regions(x2, y2, clf=svm)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
