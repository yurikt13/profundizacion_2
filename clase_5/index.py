import pandas as pd
import xlsxwriter
import sqlite3
import matplotlib.pyplot as plt
import numpy as np


# df = pd.read_csv('data.csv')
# # print(df)
# # print(df.to_string()) 

# df2 = pd.read_json('data.json')

# # print(df2.to_string())

# expenses = (
#     ['Rent', 1000],
#     ['Gas', 100],
#     ['Food', 300],
#     ['Gym', 50]
# )

# workbook = xlsxwriter.Workbook('Expenses01.xlsx')
# worksheet = workbook.add_worksheet('MyData')
# row = 0
# col = 0
# for item, cost in (expenses):
#     worksheet.write(row, col, item)
#     worksheet.write(row, col + 1, cost)
#     row += 1

# worksheet.write(row, 0, 'Total')
# worksheet.write(row, 1, '=SUM(B1:B4)')

# workbook.close()


# df_excel_2 = pd.read_excel('Expenses01.xlsx')
# print(df_excel_2)

#SQLite
#crear tabla
# connection = sqlite3.connect('sample.bd')
# cursor = connection.cursor()

# cursor.execute('''
# CREATE TABLE users (
#     id integer primary key,
#     name text not null,
#     age integer
# )
# ''')

# connection.commit()
# connection.close()


#insertar datos
# connection = sqlite3.connect('sample.bd')
# cursor = connection.cursor()

# cursor.execute('insert into users (name, age) values (?, ?)', ('Lucian', 27))

# connection.commit()
# connection.close()


#mostrar datos
# connection = sqlite3.connect('sample.bd')
# cursor = connection.cursor()

# cursor.execute("select * from users where age > ?", (55,))
# rows = cursor.fetchall()

# for row in rows:
#     print(row)

# connection.close()


# xpoints = np.array([0, 6])
# ypoints = np.array([0, 250])

# plt.plot(xpoints, ypoints)
# plt.show()



# xpoints = np.array([1, 8])
# ypoints = np.array([3, 10])

# plt.plot(xpoints, ypoints)
# plt.show()


# xpoints = np.array([1, 8])
# ypoints = np.array([3, 10])

# plt.plot(xpoints, ypoints, 'y*')
# plt.show()


# xpoints = np.array([1, 2, 6, 8])
# ypoints = np.array([3, 8, 1, 10])

# plt.plot(xpoints, ypoints)
# plt.show()


# ypoints = np.array([3, 8, 1, 10, 5, 7])

# plt.plot(ypoints)
# plt.show()


# ypoints = np.array([3, 8, 1, 10])

# plt.plot(ypoints, marker = 'o')
# plt.show()


# ypoints = np.array([3, 8, 1, 10])

# plt.plot(ypoints, 'o:r')
# plt.show()



# ypoints = np.array([3, 8, 1, 10])

# plt.plot(ypoints, marker = 'o', ms = 20)
# plt.show()


# ypoints = np.array([3, 8, 1, 10])

# plt.plot(ypoints, marker = 'o', ms = 20, mec = 'r', mfc = 'r')
# plt.show()



# ypoints = np.array([3, 8, 1, 10])

# plt.plot(ypoints, marker = 'o', ms = 20, mfc = 'r')
# plt.show()



# ypoints = np.array([3, 8, 1, 10])

# plt.plot(ypoints, marker = 'o', ms = 20, mec = '#ffAF10', mfc = '#4CAF50')
# plt.show()


# ypoints = np.array([3, 8, 1, 10])

# plt.plot(ypoints, linestyle = 'dotted')
# plt.show()



# x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
# y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

# plt.plot(x, y)

# plt.xlabel("Average Pulse")
# plt.ylabel("Calorie Burnage")

# plt.show()



x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

plt.title("Sports Watch Data")
plt.xlabel("Average Pulse")
plt.ylabel("Calorie Burnage")

plt.plot(x, y)

plt.grid()

plt.show()