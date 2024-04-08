# Importando librerias necesarias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans



X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)



# Crear el modelo K-Means con 4 clusters
kmeans = KMeans(n_clusters=4)

# Entrenar el modelo con los datos
kmeans.fit(X)

# Obtener las etiquetas de los clusters y los centroides
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Visualizar los clusters y los centroides
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering')
plt.show()