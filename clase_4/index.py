# serie -> ordena los deatos de una lista en un columna
import pandas as pd

# a = [1, 7, 2]

# myvar = pd.Series(a)

# print(myvar)

# labels -> cuando no se especifica nada los valores de la lista son indexados por números
# print(myvar[0])

# así se puede personalizar los labels:
# a = [1, 7, 2]

# myvar = pd.Series(a, index = ["x", "y", "z"])
# print(myvar)
# print(myvar["z"])


# serie con dicts-> coleccion de datos en forma etiqueta-valor
# calories = {"day1": 420, "day2": 380, "day3": 390}

# myvar = pd.Series(calories)

# print(myvar)
# print(myvar["day1"])

# calories = {"day1": 420, "day2": 380, "day3": 390}

# myvar = pd.Series(calories, index = ["day1", "day2"])

# print(myvar)


# DataFrame -> tabla completa
# data = {
#   "calories": [420, 380, 390],
#   "duration": [50, 40, 45]
# }

# myvar = pd.DataFrame(data)

# print(myvar)

# Ejercicio
# data = {
#     "Nombre" : ["Luis", "Daniel", "Mario", "Sofia"],
#     "Edad" : [30, 19, 40, 14],
#     "Genero" :  ["Masculino", "Masculino", "Masculino", "Femenino"],
#     "Peso" : ["90kg", "70kg", "50kg", "40kg"],
#     "Altura" : ["1.90", "1.60", "1.80", "1.40"]
# }

# people = pd.DataFrame(data)
# print(people)

# data = {
#   "calories": [420, 380, 390],
#   "duration": [50, 40, 45]
# }

# #load data into a DataFrame object:
# df = pd.DataFrame(data)

# print(df)

# #Locate row -> retornar filas específicas
# #refer to the row index:
# # print(df.loc[0])

# #use a list of indexes:
# df2 = df.loc[[0, 1]] #podemos crear nuevos DataFrame según el original
# print(df2)

# data = {
#   "calories": [420, 380, 390],
#   "duration": [50, 40, 45]
# }

# df = pd.DataFrame(data, index = ["day1", "day2", "day3"])

# print(df)


# Creando un DataFrame
columns = ['name', 'age', 'gender', 'job']
user1 = pd.DataFrame([['alice', 19, "F", "student"],
                      ['john', 26, "M", "student"]],
                     columns=columns)
user2 = pd.DataFrame([['eric', 22, "M", "student"],
                      ['paul', 58, "F", "manager"]],
                     columns=columns)
user3 = pd.DataFrame(dict(name=['peter', 'julie'],
                          age=[33, 44], gender=['M', 'F'],
                          job=['engineer', "scientist"]))

# print(user1)
# print(user2)
# print(user3)

#concatenar dataframes con append
# print(user1.append(user2))

#concatenar dataframes con concat
users = pd.concat([user1, user2, user3])
# print(users)

user4 = pd.DataFrame(dict(name=['alice', 'jhon', 'eric', 'julie'],
                          height = [165, 180, 175, 171]))
# print(user4)

#union de 2 dataframes
merge_inter = pd.merge(users, user4, on="name")
# print(merge_inter)

#union con claves desde ambos cuadros
users = pd.merge(users, user4, on="name", how="outer")
# print(users)

#Reorganizar por pivote ancho-largo
staked = pd.melt(users, id_vars="name", var_name="variable", value_name="value")
# print(staked)

#formato largo-ancho
print(staked.pivot(index="name", columns="variable", values="value", index=["A", "B", "C", "D", "E", "F", "G"]))