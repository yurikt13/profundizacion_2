import numpy as np
import matplotlib.pyplot as plt


data = {
    1 : 0.5,
    2 : 2.5,
    3 : 2.0,
    4 : 4.0,
    5 : 3.5,
    6 : 6.0,
    7 : 5.5
}
#print(type(data))

#1
n = len(list(data))
#print(n)

#2
xi = list(data.keys())
yi = list(data.values())
xi_array = np.array(xi)
yi_array = np.array(yi)
product_xi_yi = xi_array * yi_array
sum_xi_yi = np.sum(product_xi_yi)
print(sum_xi_yi)

#3
xi_2 = np.array([x ** 2 for x in xi])
sum_xi_2 = np.sum(xi_2)
print(sum_xi_2)

#4
sum_xi = np.sum(xi_array)
print(sum_xi)

#5
media_x = np.mean(xi_array)
print('Media de X:', int(media_x))

#6
sum_y = np.sum(yi_array)
print('Sumatoria de Y', int(sum_y))

#7
media_y = np.mean(yi_array)
print('Media de Y:', media_y)

#a1
a1 = ((n*sum_xi_yi)-(sum_xi*sum_y))/((n*sum_xi_2)-(sum_xi)**2)
print('Resultado a1: ', a1)

#a0
a0 = media_y - (a1*media_x)
print('Resultado a0: ', a0)

f8 = a0 + (a1*8)
print('Resultado f(8): ', f8)


# plt.plot(yi_array, label='Valores de x', color='purple')
# plt.plot(xi_array, label='Valores de y', color='blue')

# plt.show()




