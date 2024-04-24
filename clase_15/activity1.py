# Backpropagation with PyTorch Tensors

# Importando librerias necesarias
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.model_selection
import torch
from sklearn.preprocessing import StandardScaler

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
scalerx,scalery=StandardScaler(),StandardScaler()
X_iris=scalerx.fit_transform(X_iris)
Y_iris=StandardScaler().fit_transform(Y_iris)
#Splittraintest
X_iris_tr,X_iris_val,Y_iris_tr,Y_iris_val,label_iris_tr,label_iris_val=\
sklearn.model_selection.train_test_split(X_iris,Y_iris,label_iris,train_size=0.5, stratify=label_iris)
# plt.show()

# X=X_iris_tr; Y=Y_iris_tr; X_val=X_iris_val; Y_val=Y_iris_val
def two_layer_regression_tensor_train(X, Y, X_val, Y_val, lr, nite):
  dtype = torch.float
  device = torch.device("cpu")
  # device = torch.device("cuda:0") # Uncomment this to run on GPU
  # N is batch size; D_in is input dimension;
  # H is hidden dimension; D_out is output dimension.
  N, D_in, H, D_out = X.shape[0], X.shape[1], 100, Y.shape[1]
  # Create random input and output data
  X = torch.from_numpy(X)
  Y = torch.from_numpy(Y)
  X_val = torch.from_numpy(X_val)
  Y_val = torch.from_numpy(Y_val)
  # Randomly initialize weights
  W1 = torch.randn(D_in, H, device=device, dtype=dtype)
  W2 = torch.randn(H, D_out, device=device, dtype=dtype)
  losses_tr, losses_val = list(), list()
  learning_rate = lr
  for t in range(nite):
    # Forward pass: compute predicted y
    z1 = X.mm(W1)
    h1 = z1.clamp(min=0)
    y_pred = h1.mm(W2)
    # Compute and print loss
    loss = (y_pred- Y).pow(2).sum().item()
    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred- Y)
    grad_w2 = h1.t().mm(grad_y_pred)
    grad_h1 = grad_y_pred.mm(W2.t())
    grad_z1 = grad_h1.clone()
    grad_z1[z1 < 0] = 0
    grad_w1 = X.t().mm(grad_z1)
    # Update weights using gradient descent
    W1-= learning_rate * grad_w1
    W2-= learning_rate * grad_w2
    # Forward pass for validation set: compute predicted y
    z1 = X_val.mm(W1)
    h1 = z1.clamp(min=0)
    y_pred_val = h1.mm(W2)
    loss_val = (y_pred_val- Y_val).pow(2).sum().item()
    losses_tr.append(loss)
    losses_val.append(loss_val)
    if t % 10 == 0:
      print(t, loss, loss_val)
  return W1, W2, losses_tr, losses_val

W1, W2, losses_tr, losses_val = two_layer_regression_tensor_train(X=X_iris_tr, Y=Y_iris_tr, X_val=X_iris_val, Y_val=Y_iris_val,lr=1e-4, nite=50)

# Crear una nueva figura para la segunda gráfica
plt.figure()

# Trazar la segunda gráfica
plt.plot(np.arange(len(losses_tr)), losses_tr, "-b", np.arange(len(losses_val)), losses_val, "-r")
plt.grid()
plt.show()