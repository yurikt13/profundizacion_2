# Importando librerias necesarias
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.model_selection
from sklearn.preprocessing import StandardScaler

iris=sns.load_dataset("iris")
# print(iris)

df=iris[iris.species!="setosa"]

g=sns.pairplot(df,hue="species")
plt.show()

df['species_n'] = iris.species.map({'versicolor':1, 'virginica':2})

#Y= 'petal_length', 'petal_width'; X= 'sepal_length','sepal_width')
X_iris=np.asarray(df.loc[:,['sepal_length', 'sepal_width']],dtype=np.float32)
Y_iris=np.asarray(df.loc[:,['petal_length', 'petal_width']],dtype=np.float32)
label_iris=np.asarray(df.species_n,dtype=int)

#Scale
scalerx,scalery=StandardScaler(),StandardScaler()
X_iris=scalerx.fit_transform(X_iris)
Y_iris=StandardScaler().fit_transform(Y_iris)
# print(X_iris, Y_iris)

#Split train test
X_iris_tr,X_iris_val,Y_iris_tr,Y_iris_val,label_iris_tr,label_iris_val=\
sklearn.model_selection.train_test_split(X_iris,Y_iris,label_iris,train_size=0.5,stratify=label_iris)
print(X_iris_tr, Y_iris_tr)