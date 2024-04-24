# Backpropagation with PyTorch optim

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

def two_layer_regression_nn_optim_train(X, Y, X_val, Y_val, lr, nite):
 # N is batch size; D_in is input dimension;
 # H is hidden dimension; D_out is output dimension.
 N, D_in, H, D_out = X.shape[0], X.shape[1], 100, Y.shape[1]
 X = torch.from_numpy(X)
 Y = torch.from_numpy(Y)
 X_val = torch.from_numpy(X_val)
 Y_val = torch.from_numpy(Y_val)
 # Use the nn package to define our model and loss function.
 model = torch.nn.Sequential(
 torch.nn.Linear(D_in, H),
 torch.nn.ReLU(),
 torch.nn.Linear(H, D_out),
 )
 loss_fn = torch.nn.MSELoss(reduction='sum')
 losses_tr, losses_val = list(), list()
 # Use the optim package to define an Optimizer that will update the weights of
 # the model for us. Here we will use Adam; the optim package contains many other
 # optimization algoriths. The first argument to the Adam constructor tells the
 # optimizer which Tensors it should update.
 learning_rate = lr
 optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
 for t in range(nite):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(X)
    # Compute and print loss.
    loss = loss_fn(y_pred, Y)
    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()
    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()
    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()
    with torch.no_grad():
      y_pred = model(X_val)
      loss_val = loss_fn(y_pred, Y_val)

    if t % 10 == 0:
      print(t, loss.item(), loss_val.item())

    losses_tr.append(loss.item())
    losses_val.append(loss_val.item())

 return model, losses_tr, losses_val

model, losses_tr, losses_val = two_layer_regression_nn_optim_train(X=X_iris_tr, Y=Y_iris_tr, X_val=X_iris_val, Y_val=Y_iris_val,lr=1e-3, nite=50)

# Crear una nueva figura para la segunda gráfica
plt.figure()

# Trazar la segunda gráfica
plt.plot(np.arange(len(losses_tr)), losses_tr, "-b", np.arange(len(losses_val)), losses_val, "-r")
plt.grid()
plt.show()