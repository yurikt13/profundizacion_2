import numpy as np

data1 = [1, 2, 3, 4, 5] #lista
# print(data1)

arr1 = np.array(data1) #arreglo de una dimensión(vector)
# print(arr1)

data2 = [range(1, 5), range(5, 9)] #lista de listas
# print(data2)

arr2 = np.array(data2) #arreglo de dos dimensiones(matriz)
# print(arr2)

arr3 = arr2.tolist()
# print(arr3)


#Arreglos especiales
x = zeros_uno = np.zeros(10) #arreglo 10 ceros
# print(x)

y = np.zeros((3, 6)) #arreglo de ceros 3x6
# print(y)

xx = np.ones(10) #arreglo de diez unos
# print(xx)

yy = np.linspace(0, 1, 5) #arreglo con inicio, fin, cinco puntos lineales
# print(yy)

zz = np.logspace(0, 4, 10) #arreglo iniciando en cero con diez números espacioados en logaritmo
# print(zz)

int_array = np.arange(5) #array de 5 enteros iniciando en 0
# print(int_array)

float_array = int_array.astype(float) #convierte los datos de int a float
# print(float_array)


#Examinando arreglos
a = arr1.dtype
# print(a)

b = arr2.dtype #tipo de datos del arreglo
# print(b)

c = arr2.ndim #dimensiones del arreglo
# print(c)

d = arr2.shape #forma del arreglo
# print(d)

e = arr2.size #número de elementos en el arreglo
# print(e)

f = len(arr2) #longitud del areglo(2-> tamaño de la primera dimensión)
# print(f)


#Reconfigurando un arreglo
arr = np.arange(10) #arreglo de 10 enteros
# print(arr)

arr = np.arange(10, dtype=float) #convierte el arreglo en float
# print(arr)

arr = np.arange(10, dtype=float).reshape((2, 5)) #lo cambia de 1x10 a 2x5
# print(arr)
# print(arr.shape) #nueva forma
# print(arr.reshape(5, 2)) #arreglo en 5x2 sin modificarlo
# print(arr)


#Añadir un eje
a = np.array([0, 1])
# print(a)

#forma #1
a_col = a[:, np.newaxis]
# print(a_col)

#forma #2
a_col = a[:, None]
# print(a_col)

#Calculado transpuesta
# print(a_col.T)


#Flatten -> siempre retorna una copia plana del arreglo original
arr = np.arange(10)
# print(arr)

#reshape -> cambiar la forma de un arreglo multidimensional sin cambiar sus datos
arr = np.arange(10).reshape((2, 5))
# print(arr)

arr_flt = arr.flatten()
# print(arr_flt)

arr_flt[0] = 33
# print(arr_flt)
# print(arr)


def standardize(X):
    media_cols = np.mean(X, axis=0)
    std_cols = np.std(X, axis=0)
    standardized_X = (X - media_cols) / std_cols
    return standardized_X


X = np.random.randn(4, 2)
print("Arreglo original:")
print(X)

# Para cada columna, encuentra el índice de la fila que posee el mínimo valor
indices_minimos = np.argmin(X, axis=0)
print('Índice mínimo de cada columna:', indices_minimos)

# Utilizar la función standardize
datos_estandarizados = standardize(X)
print('Columnas centradas y escaladas:')
print(datos_estandarizados)