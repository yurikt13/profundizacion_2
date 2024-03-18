import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv('data_activity.csv')
print(df)

# Crear la gráfica de dispersión de los datos
df.plot.scatter(x="E", y="P", label="Datos Originales")

# Ajustar una regresión lineal a los datos
reg = LinearRegression()
reg.fit(df[["E"]], df["P"])

# Imprimir los coeficientes de la regresión lineal
print("Intercepto:", reg.intercept_)
print("Coeficiente de la pendiente:", reg.coef_[0])

# Predecir el peso para una estatura específica
estaturas_nuevas = [[1.50], [1.55], [1.60], [1.65], [1.70], [1.75], [1.80], [1.85], [1.90]]
predicciones = reg.predict(estaturas_nuevas)
print("Estaturas Nuevas:", estaturas_nuevas)
print("Predicciones de Peso:", predicciones)

# Agregar la línea de regresión a la gráfica
plt.plot(estaturas_nuevas, predicciones, color='red', label="Regresión Lineal")

# Mostrar la gráfica con los datos y la regresión lineal
plt.xlabel("Estatura")
plt.ylabel("Peso")
plt.legend()
plt.title("Relación entre Estatura y Peso")
plt.grid(True)
plt.show()