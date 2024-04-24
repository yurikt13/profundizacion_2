# K-Vecinos más cercanos (KNN)
# Datos de clientes bancarios:crédito

# Importando librerias necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

# Leyendo archivo data a trabajar
clientes = pd.read_csv('creditos.csv')
# print(clientes)

# Pagadores VS Deudores
buenos = clientes[clientes["cumplio"]==1]
malos = clientes[clientes["cumplio"]==0]
# print(buenos, malos)

# Preparación de los datos (Escalar)
datos = clientes[["edad", "credito"]]
clase = clientes["cumplio"]

escalador = preprocessing.MinMaxScaler()

datos = escalador.fit_transform(datos)
# print(datos)

# Creación del modelo KNN
# Valor de K
clasificador = KNeighborsClassifier(n_neighbors=3)
clasificador.fit(datos, clase)

# Nuevo solicitante (Clasificación)
edad = 60
monto = 350000

# Escalar datos del nuevo solicitante
solicitante = escalador.transform([[edad, monto]])
print("Clase", clasificador.predict(solicitante))
print("Probabilidades por clase", clasificador.predict_proba(solicitante))

#calcular clase y probabilidades

# Graficar
# Crear una nueva figura para combinar ambas gráficas
plt.figure(figsize=(21, 6))

# Primera gráfica: Pagadores VS Deudores
plt.subplot(1, 3, 1)
plt.scatter(buenos["edad"], buenos["credito"], marker="*", s=150, color="skyblue", label="Sí pagó (Clase #1)")
plt.scatter(malos["edad"], malos["credito"], marker="*", s=150, color="red", label="No pagó (Clase #0)")
plt.ylabel("Monto del crédito")
plt.xlabel("Edad")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title('Pagadores VS Deudores')

# Segunda gráfica: Nuevo solicitante
plt.subplot(1, 3, 2)
plt.scatter(buenos["edad"], buenos["credito"], marker="*", s=150, color="skyblue", label="Sí pagó (Clase #1)")
plt.scatter(malos["edad"], malos["credito"], marker="*", s=150, color="red", label="No pagó (Clase #0)")
plt.scatter(edad, monto, marker="P", s=250, color="green", label="Solicitante")
plt.ylabel("Monto del crédito")
plt.xlabel("Edad")
plt.legend(bbox_to_anchor=(1, 0.3))
plt.title('Nuevo solicitante')

# Regiones de las clases
# Defiir regiones de las clases entre los Pagadores VS y No pagadores

# Datos sintéticos de todos los posibles solicitantes
creditos = np.array([np.arange(100000, 600010, 1000)] * 43).reshape(1, -1)
edades = np.array([np.arange(18, 61)] * 501).reshape(1, -1)
todos = pd.DataFrame(np.stack((edades, creditos), axis=2)[0], columns=["edad", "credito"])

# Escalar datos
solicitantes = escalador.transform(todos)

# Predecir todas las clases
clases_resultantes = clasificador.predict(solicitantes)

#Codigo para graficar 3
plt.subplot(1, 3, 3)
buenos = todos[clases_resultantes==1]
malos = todos[clases_resultantes==0]
plt.scatter(buenos["edad"], buenos["credito"], marker="*", s=150, color="skyblue", label="Sí pagará (Clase #1)")
plt.scatter(malos["edad"], malos["credito"], marker="*", s=150, color="red", label="No pagará (Clase #0)")
plt.ylabel("Monto del crédito")
plt.xlabel("Edad")
plt.legend(bbox_to_anchor=(1, 0.2))
plt.title('Regiones de las clases (Nuevo solicitante)')

plt.tight_layout()  # Ajustar automáticamente los elementos de la gráfica para evitar superposiciones
plt.show()


