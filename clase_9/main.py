# Importando librerias necesarias
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA



# Importando los datos a ser analizados
df = load_wine(as_frame=True)
df = df.frame
df.head()



# Se elimina la(s) columnas que no contengan características relevantes
df.drop('target', axis =1, inplace=True)

# Check the data informations
df.info()



# Normalizando los datos (Por medio de un escalado)
scaler =StandardScaler()

features =scaler.fit(df)
features =features.transform(df)

# Convert to pandas Dataframe
scaled_df =pd.DataFrame(features,columns=df.columns)
# Print the scaled data
scaled_df.head(2)


# Quitando los labels(etiquetas) que son la primera fila
X=scaled_df.values



# Aplicando método elbow para obtener el número optimo de clusters
wcss = {}
for i in range(1, 11):
	kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
	kmeans.fit(X)
	wcss[i] = kmeans.inertia_

plt.plot(wcss.keys(), wcss.values(), 'gs-')
plt.xlabel("Values of 'k'")
plt.ylabel('WCSS')
# plt.show()



# Teniendo el número optimo de cluster se aplica el algoritmo k-means
kmeans=KMeans(n_clusters=3)
kmeans.fit(X)



# Obtener array de los centros de cada cluster
kmeans.cluster_centers_



# Etiquetar índices de los cluster de acuerdo a la muestra a la cual pertenecen
kmeans.labels_
# print(kmeans.labels_)



# Aplicar la técnica PCA para la reducción de dimensionalidad:
# Análisis de componentes principales
pca=PCA(n_components=2)

reduced_X=pd.DataFrame(data=pca.fit_transform(X),columns=['PCA1','PCA2'])

#Reduced Features
reduced_X.head()



# Reducir los centros de los cluster usando PCA
centers=pca.transform(kmeans.cluster_centers_)

# reduced centers
centers



# Representar los clusters
plt.figure(figsize=(7,5))

# Scatter plot
plt.scatter(reduced_X['PCA1'],reduced_X['PCA2'],c=kmeans.labels_)
plt.scatter(centers[:,0],centers[:,1],marker='x',s=100,c='red')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('Wine Cluster')
plt.tight_layout()
# plt.show()



# Reducir número de componentes principales
pca.components_


# Representandos nuevas componentes de PCA
component_df=pd.DataFrame(pca.components_,index=['PCA1',"PCA2"],columns=df.columns)
# Heat map
sns.heatmap(component_df)
plt.show()