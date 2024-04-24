# Importando librerias necesarias
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Definir parámetros de la distribución normal
mu = 4 #media
varianza = 1 #varianza
sigma = np.sqrt(varianza) #desviación estandar

# Generar valores de x
x = np.linspace(mu-3*varianza, mu+3*varianza, 100)

# Graficar la función de densidad de probabilidad (PDF) de la distribución normal
plt.plot(x, norm.pdf(x, mu, sigma))
print(x)

# Título y etiquetas de los ejes
plt.title('Distribución Normal')
plt.xlabel('Valores de x')
plt.ylabel('Densidad de probabilidad')
plt.show()