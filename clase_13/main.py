# Importando librerias necesarias
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Datos de entrenamiento
celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float) # Datos de entrada
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float) # Salidas deseadas
# Nuestro objetivo es crear una red neuronal que nos realice esta conversión automáticamente

# capa = tf.keras.layers.Dense(units=1, input_shape=[1]) #se construye el modelo con una nreurona(Perceptron)
# modelo = tf.keras.Sequential([capa]) # una sola capa

oculta1 = tf.keras.layers.Dense(units = 4, input_shape = [1])
oculta2 = tf.keras.layers.Dense(units = 3)
salida = tf.keras.layers.Dense(units = 1)
modelo = tf.keras.Sequential([
    # tf.keras.layers.Input(shape=(1,)),
    oculta1, 
    oculta2, 
    salida
])

# # Se calcula el error como el error cuadrático
modelo.compile(
    optimizer = tf.keras.optimizers.Adam(0.1),
    loss = 'mean_squared_error')

print("Comenzando entrenamiento..")
#COmenzamos a ajustar los pesos
historial = modelo.fit(celsius, fahrenheit, epochs = 100, verbose = False) #Se hace el entrenamiento con mil pasadas (por ejemplo)
print("Modelo entrenado")

plt.xlabel("# Epoca")
plt.ylabel("Magnitud del error")
plt.plot(historial.history["loss"])
plt.show()

print("haciendo una prediccion")
resultado = modelo.predict(np.array([100.0]))
print("El resultado es: " + str(resultado) + " fahrenheit")

print("Variables internas del modelo")
# print(capa.get_weights())
print(oculta1.get_weights())
print(oculta2.get_weights())
print(salida.get_weights())