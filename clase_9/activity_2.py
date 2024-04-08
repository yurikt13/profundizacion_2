# Importando librerias necesarias
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Generamos valores aleatorios para las características (features) X1, X2 y X3
np.random.seed(0)  # Esto asegura que los resultados sean reproducibles
num_samples = 300
X1 = np.random.normal(loc=5, scale=2, size=num_samples)
X2 = np.random.normal(loc=3, scale=1.5, size=num_samples)
X3 = np.random.normal(loc=10, scale=3, size=num_samples)

# Creamos el DataFrame
data = {
    'X1': X1,
    'X2': X2,
    'X3': X3
}

df = pd.DataFrame(data)
# print(df)


# Creamos el modelo KMeans con 2 clusters
kmeans = KMeans(n_clusters=4)

# Ajustamos el modelo a los datos
kmeans.fit(df)

# Obtenemos las etiquetas de los clusters
labels = kmeans.labels_

# Añadimos las etiquetas al DataFrame
df['Cluster'] = labels

# Mostramos los resultados
print(df)


# Visualizamos los clusters en un gráfico 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['X1'], df['X2'], df['X3'], c=df['Cluster'], cmap='viridis')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
ax.set_title('KMeans Clustering')
plt.show()