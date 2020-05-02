import numpy as np
from sklearn import neighbors ,datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data[:,:2] 
Y = iris.target
print(Y)
knn = neighbors.KNeighborsClassifier(n_neighbors=9)
knn.fit(X,Y)
plt.scatter(X[:,0], X[:,1],c =Y) 
plt.show()

knn.predict(np.array([[5.0,2.3]])) 