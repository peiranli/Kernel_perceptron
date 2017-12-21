w1, b1 = quadratic_kernel(x1, y1)
w2, b2 = quadratic_kernel(x2, y2)


delta = 0.005
x1_min, x1_max = 0, 11
x2_min, x2_max = 0, 11
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, delta), np.arange(x2_min, x2_max, delta))
Z = z_func(xx1.ravel(), xx2.ravel(), w1, b1)
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
Z = z_func(xx3.ravel(), xx4.ravel(), w2, b2)
Z = Z.reshape(xx3.shape)
plt.pcolormesh(xx3, xx4, Z, cmap=plt.cm.Pastel2,vmin=0, vmax=2)
plt.scatter(x2_1[:,0],x2_1[:,1],color='red',marker='>')
plt.scatter(x2_2[:,0],x2_2[:,1],color='blue',marker='o')
plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.show()

alpha1, b1 = rbf_kernel(x1, y1, 100)
alpha2, b2 = rbf_kernel(x2, y2, 100)

delta = 0.01
x1_min, x1_max = 0, 11
x2_min, x2_max = 0, 11
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, delta), np.arange(x2_min, x2_max, delta))
Z = predict(xx1.ravel(), xx2.ravel(), alpha1, b1, x1, y1)
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
Z = predict(xx3.ravel(), xx4.ravel(), alpha2, b2, x2, y2)
Z = Z.reshape(xx3.shape)
plt.pcolormesh(xx3, xx4, Z, cmap=plt.cm.Pastel2,vmin=0, vmax=2)
plt.scatter(x2_1[:,0],x2_1[:,1],color='red',marker='>')
plt.scatter(x2_2[:,0],x2_2[:,1],color='blue',marker='o')
plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.show()
