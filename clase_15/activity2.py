# Backpropagation with PyTorch: Tensors and autograd

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
# df
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
# del X, Y, X_val, Y_val
def two_layer_regression_autograd_train(X, Y, X_val, Y_val, lr, nite):
 dtype = torch.float
 device = torch.device("cpu")
 # device = torch.device("cuda:0") # Uncomment this to run on GPU
 # N is batch size; D_in is input dimension;
 # H is hidden dimension; D_out is output dimension.
 N, D_in, H, D_out = X.shape[0], X.shape[1], 100, Y.shape[1]
 # Setting requires_grad=False indicates that we do not need to compute gradients
 # with respect to these Tensors during the backward pass.
 X = torch.from_numpy(X)
 Y = torch.from_numpy(Y)
 X_val = torch.from_numpy(X_val)
 Y_val = torch.from_numpy(Y_val)
 # Create random Tensors for weights.
 # Setting requires_grad=True indicates that we want to compute gradients with
 # respect to these Tensors during the backward pass.
 W1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
 W2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

 losses_tr, losses_val = list(), list()
 learning_rate = lr
 for t in range(nite):
  # Forward pass: compute predicted y using operations on Tensors; these
  # are exactly the same operations we used to compute the forward pass using
  # Tensors, but we do not need to keep references to intermediate values since
  # we are not implementing the backward pass by hand.
  y_pred = X.mm(W1).clamp(min=0).mm(W2)
  # Compute and print loss using operations on Tensors.
  # Now loss is a Tensor of shape (1,)
  # loss.item() gets the scalar value held in the loss.
  loss = (y_pred- Y).pow(2).sum()
  # Use autograd to compute the backward pass. This call will compute the
  # gradient of loss with respect to all Tensors with requires_grad=True.
  # After this call w1.grad and w2.grad will be Tensors holding the gradient
  # of the loss with respect to w1 and w2 respectively.
  loss.backward()
  # Manually update weights using gradient descent. Wrap in torch.no_grad()
  # because weights have requires_grad=True, but we don't need to track this
  # in autograd.
  # An alternative way is to operate on weight.data and weight.grad.data.
  # Recall that tensor.data gives a tensor that shares the storage with
  # tensor, but doesn't track history.
  # You can also use torch.optim.SGD to achieve this.
  with torch.no_grad():
    W1-= learning_rate * W1.grad
    W2-= learning_rate * W2.grad
    # Manually zero the gradients after updating weights
    W1.grad.zero_()
    W2.grad.zero_()
    y_pred = X_val.mm(W1).clamp(min=0).mm(W2)
    # Compute and print loss using operations on Tensors.
    # Now loss is a Tensor of shape (1,)
    # loss.item() gets the scalar value held in the loss.
    loss_val = (y_pred- Y).pow(2).sum()

  if t % 10 == 0:
    print(t, loss.item(), loss_val.item())

  losses_tr.append(loss.item())
  losses_val.append(loss_val.item())

 return W1, W2, losses_tr, losses_val


W1, W2, losses_tr, losses_val = two_layer_regression_autograd_train(X=X_iris_tr, Y=Y_iris_tr, X_val=X_iris_val, Y_val=Y_iris_val,lr=1e-4, nite=50)

# Crear una nueva figura para la segunda gráfica
plt.figure()

# Trazar la segunda gráfica
plt.plot(np.arange(len(losses_tr)), losses_tr, "-b", np.arange(len(losses_val)), losses_val, "-r")
plt.grid()
plt.show()