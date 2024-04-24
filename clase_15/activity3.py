# Backpropagation with PyTorch: nn

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
def two_layer_regression_nn_train(X, Y, X_val, Y_val, lr, nite):
 # N is batch size; D_in is input dimension;
 # H is hidden dimension; D_out is output dimension.
 N, D_in, H, D_out = X.shape[0], X.shape[1], 100, Y.shape[1]
 X = torch.from_numpy(X)
 Y = torch.from_numpy(Y)
 X_val = torch.from_numpy(X_val)
 Y_val = torch.from_numpy(Y_val)
 # Use the nn package to define our model as a sequence of layers. nn.Sequential
 # is a Module which contains other Modules, and applies them in sequence to
 # produce its output. Each Linear Module computes output from input using a
 # linear function, and holds internal Tensors for its weight and bias.
 model = torch.nn.Sequential(
 torch.nn.Linear(D_in, H),
 torch.nn.ReLU(),
 torch.nn.Linear(H, D_out),
 )
 # The nn package also contains definitions of popular loss functions; in this
 # case we will use Mean Squared Error (MSE) as our loss function.
 loss_fn = torch.nn.MSELoss(reduction='sum')
 losses_tr, losses_val = list(), list()
 learning_rate = lr
 for t in range(nite):
  # Forward pass: compute predicted y by passing x to the model. Module objects
  # override the __call__ operator so you can call them like functions. When
  # doing so you pass a Tensor of input data to the Module and it produces
  # a Tensor of output data.
  y_pred = model(X)
  # Compute and print loss. We pass Tensors containing the predicted and true
  # values of y, and the loss function returns a Tensor containing the
  # loss.
  loss = loss_fn(y_pred, Y)
  # Zero the gradients before running the backward pass.
  model.zero_grad()
  # Backward pass: compute gradient of the loss with respect to all the learnable
  # parameters of the model. Internally, the parameters of each Module are stored
  # in Tensors with requires_grad=True, so this call will compute gradients for
  # all learnable parameters in the model.
  loss.backward()
  # Update the weights using gradient descent. Each parameter is a Tensor, so
  # we can access its gradients like we did before.
  with torch.no_grad():
    for param in model.parameters():
      param-=learning_rate*param.grad
      y_pred=model(X_val)
      loss_val=(y_pred-Y_val).pow(2).sum()

  if t%10==0:
    print(t,loss.item(),loss_val.item())

  losses_tr.append(loss.item())
  losses_val.append(loss_val.item())

 return model,losses_tr,losses_val


model,losses_tr,losses_val=two_layer_regression_nn_train(X=X_iris_tr,Y=Y_iris_tr,X_val=X_iris_val,Y_val=Y_iris_val,lr=1e-4,nite=50)

# Crear una nueva figura para la segunda gráfica
plt.figure()

# Trazar la segunda gráfica
plt.plot(np.arange(len(losses_tr)), losses_tr,"-b",np.arange(len(losses_val)),losses_val,"-r")
plt.grid()
plt.show()