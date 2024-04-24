# Importando librerias necesarias
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Definir valores de variables
data = {
    'Age': [35, 45, 28, 60, 50],
    'Frequency of visits': [2, 1, 3, 1, 2],
    'Pain level': [3, 7, 5, 4, 6],
    'Compliance': [90, 75, 80, 85, 70]
}

df = pd.DataFrame(data)

#Escalar los datos
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Aplicar algoritmo K-Means
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(scaled_data)

# Agregar las etiquetas de los clusters al df
df['Cluster'] = kmeans.labels_

# Visualizar
plt.scatter(df['Age'], df['Pain level'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Age')
plt.ylabel('Pain level')
plt.title('Patient Clusters')
plt.show()