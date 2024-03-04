import pandas as pd
import matplotlib.pyplot as plt

# leyendo datos
df = pd.read_csv('data.csv')

# seleccionando datos de columna Pulse
pulse_data = df['Pulse']

# filtro suavizado
filtered_data = [(pulse_data[i] + pulse_data[i+1]) / 2 for i in range(len(pulse_data) - 1)]

# graficando datos originales
plt.plot(pulse_data, label='Datos Originales', color='purple')

# graficando datos con filtrado suavizado
plt.plot(filtered_data, label='Datos filtrados', color='yellow')


# agregando los titulos
plt.title('Datos originales y Filtrados')
plt.xlabel('Índice de Muestra')
plt.ylabel('Pulse')
plt.legend()

plt.show()