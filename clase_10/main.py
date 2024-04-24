# Importando librerias necesarias
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

# Gráfico de puntos x, y
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(x, y)
plt.grid()
plt.title('Datos de entrada')
plt.xlabel('x')
plt.ylabel('y')
# plt.show()

# Método Elbow
data = list(zip(x,y))
# print(data)
inertias = []
for i in range(1, 11):
  kmeans = KMeans(n_clusters=i)
  kmeans.fit(data)
  inertias.append(kmeans.inertia_)

plt.subplot(1, 2, 2)
plt.plot(range(1, 11), inertias, marker='o')
plt.title('Método del codo')
plt.xlabel('Número de clusters')
plt.ylabel('Inercia')

plt.tight_layout()
# plt.show()

# Clustering final
kmeans = KMeans(n_clusters=6)
kmeans.fit(data)

plt.figure()
plt.scatter(x, y, c=kmeans.labels_)
plt.title('Clustering final')
plt.xlabel('x')
plt.ylabel('y')
plt.show()