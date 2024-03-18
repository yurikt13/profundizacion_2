import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')
# print(df)

df.plot.scatter(x = "X", y = "Y")

# plt.show()

reg = LinearRegression()
reg.fit(df[["X"]], df["Y"])

print(reg.predict([[11]]))

print(reg.predict([[3], [6]]))

print(reg.intercept_)

print(reg.coef_)

print(f"La línea intercepta en {reg.intercept_} y tiene una pendiente de: {reg.coef_}")

print("Valores obtenidos con el predictor:")
print(reg.coef_*(df["X"]+1) + reg.intercept_)

print("De otro modo:")
pred = pd.Series(reg.predict(df[["X"]]))
df["Predicción"] = pred
print(df)

ax = df.plot.line(x= "X", y= "Predicción")
df.plot.scatter(x= "X", y= "Y", ax = ax, color="#ff0000")

plt.show()