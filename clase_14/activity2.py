# Importando librerias necesarias
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.model_selection


iris=sns.load_dataset("iris")
#g=sns.pairplot(iris,hue="species")
df=iris[iris.species!="setosa"]
g=sns.pairplot(df,hue="species")
df['species_n']=iris.species.map({'versicolor':1,'virginica':2})
#Y='petal_length','petal_width';X='sepal_length','sepal_width')
X_iris=np.asarray(df.loc[:,['sepal_length','sepal_width']],dtype=np.float32)
Y_iris=np.asarray(df.loc[:,['petal_length','petal_width']],dtype=np.float32)
label_iris=np.asarray(df.species_n,dtype=int)
#Scale
from sklearn.preprocessing import StandardScaler
scalerx,scalery=StandardScaler(),StandardScaler()
X_iris=scalerx.fit_transform(X_iris)
Y_iris=StandardScaler().fit_transform(Y_iris)
#Splittraintest
X_iris_tr,X_iris_val,Y_iris_tr,Y_iris_val,label_iris_tr,label_iris_val=\
sklearn.model_selection.train_test_split(X_iris,Y_iris,label_iris,train_size=0.5, stratify=label_iris)
# plt.show()

def two_layer_regression_numpy_train(X, Y, X_val, Y_val, lr, nite):
  # N is batch size; D_in is input dimension;
  # H is hidden dimension; D_out is output dimension.
  # N, D_in, H, D_out = 64, 1000, 100, 10
  N, D_in, H, D_out = X.shape[0], X.shape[1], 100, Y.shape[1]
  W1 = np.random.randn(D_in, H)
  W2 = np.random.randn(H, D_out)
  losses_tr, losses_val = list(), list()
  learning_rate = lr
  for t in range(nite):
    # Forward pass: compute predicted y
    z1 = X.dot(W1)
    h1=np.maximum(z1,0)
    Y_pred=h1.dot(W2)
    #Computeandprintloss
    loss=np.square(Y_pred-Y).sum()
    #Backproptocomputegradientsofw1andw2withrespecttoloss
    grad_y_pred=2.0*(Y_pred-Y)
    grad_w2=h1.T.dot(grad_y_pred)
    grad_h1=grad_y_pred.dot(W2.T)
    grad_z1=grad_h1.copy()
    grad_z1[z1<0]=0
    grad_w1=X.T.dot(grad_z1)
    #Updateweights
    W1-=learning_rate*grad_w1
    W2-=learning_rate*grad_w2
    #Forwardpassforvalidationset:computepredictedy
    z1=X_val.dot(W1)
    h1=np.maximum(z1,0)
    y_pred_val=h1.dot(W2)
    loss_val=np.square(y_pred_val-Y_val).sum()
    losses_tr.append(loss)
    losses_val.append(loss_val)
    if t % 10 == 0:
      print(t,loss,loss_val)
  return W1,W2,losses_tr,losses_val

W1,W2,losses_tr,losses_val=two_layer_regression_numpy_train(X=X_iris_tr,Y=Y_iris_tr, X_val=X_iris_val,Y_val=Y_iris_val,lr=1e-4,nite=50)

# Crear una nueva figura para la segunda gráfica
plt.figure()

# Trazar la segunda gráfica
plt.plot(np.arange(len(losses_tr)), losses_tr,"-b",np.arange(len(losses_val)),losses_val,"-r")
plt.show()