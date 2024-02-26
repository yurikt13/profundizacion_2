import random

array = []


for i in range(5):
    array.append(random.randint(1,5))


arr = list(set(array))

print(arr)

cont = 0
suma = 0
for i in arr:
    suma += i
    cont += 1

media = suma / cont
print("Media: ", media)

suma_cuadrados_diff = 0
for j in arr:
    suma_cuadrados_diff += (j - media) ** 2
    print("suma_cuadrados_diff: ", suma_cuadrados_diff)

print("Desviación estandar: ", suma_cuadrados_diff)

    
