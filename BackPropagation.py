# CHAIN RULE: x --> [a(x)] --> y --> [b(y)] --> z
'''first calculate--> dy/dx , then calculate--> dz/dy
  atlast calculate final gradient: dz/dx = dy/dx * dz/dy'''

# COMPUTATIONAL GRAPH: having inputs , operator (having function) , output

# 3 important things: [1] Forward Pass : compute loss , [2] Compute local gradients, [3] Backward pass : compute dLoss/dWeights using the chain rule.
'''let, x(i/p) * w(weights)  = y' (o/p) and  y --> actual y
      [1]  loss = (y' - y)**2 = (wx - y)**2
      [2]  calculate local (intermediate) gradients at each node (dloss/ds,ds/dy',dy'/dw)
      [3]  use chain rule, dloss/ds * ds/dy' --> dloss/dy'*dy'/dw ---> dloss/dw [final gradient]

      so, finally we have to minimize loss by updating weights.'''

# JUST PRACTICE
import torch
x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0,requires_grad = True)

# forward pass and compute loss
y_hat = w * x
loss = (y_hat - y) ** 2 

print(loss)

# backward pass
loss.backward()
print(w.grad)
print('*' * 20)
## next update weights
## next fwd and bwd pass
#---------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------

# ACTUAL COMPUTATION
# 1: PREDICTION , GRADIENT COMPUTATION , LOSS COMPUTATION, PARAMETER UPDATES --> ALL MANUALLY

import numpy as np

# f = w *x
# y = 2 *x

x = np.array([1,2,3,4],dtype = np.float32)
y = np.array([2,4,6,8],dtype = np.float32)
w  = 0.0
# model prediction
def forward(x):
    return w*x   # y_pred
# loss = MSE
def loss(y,y_pred):
    return ((y_pred - y)**2).mean()
# gradient
'''J = loss = 1/N * (w*x - y)**2
   dJ/dw = 1/N * 2x * (wx - y)'''
def grad(x,y,y_pred):
    g = np.dot(2*x, (y_pred - y)).mean()
    return g
print('prediction before training: f(5) = {0}'.format(forward(5)))
print('prediction before training: f(8) = {0}'.format(forward(8))) 
# training
lr = 0.01
epoch = 10
for e in range(epoch):
    y_pred = forward(x) #PREDICTION
    l = loss(y,y_pred)  #LOSS
    dw = grad(x,y,y_pred) #GRADIENT --> dl/dw
    w -= lr * dw #WEIGHT UPDATES
    if e % 1 == 0:
        print('epoch {0}: w = {1}, loss = {2}'.format(e + 1,round(w,3),l))
print('prediction after training: f(5) = {0}'.format(forward(5)))
print('prediction after training: f(8) = {0}'.format(forward(8))) 
print('*' * 20)    

#---------------------------------------------------------------------------------------------------------------------------------------


# 2: PREDICTION [manually] , GRADIENT COMPUTATION [autograd] , LOSS COMPUTATION [manually], PARAMETER UPDATES [manually]
import torch

# f = w *x
# y = 2 *x

x = torch.tensor([1,2,3,4],dtype = torch.float32)
y = torch.tensor([2,4,6,8],dtype = torch.float32)

w  = torch.tensor(0.0,dtype = torch.float32, requires_grad = True)

# model prediction
def forward(x):
    return w*x   # y_pred
# loss = MSE
def loss(y,y_pred):
    return ((y_pred - y)**2).mean()

print('prediction before training: f(5) = {0}'.format(forward(5)))
print('prediction before training: f(8) = {0}'.format(forward(8)))

# training
lr = 0.01
epoch = 100
for e in range(epoch):
    y_pred = forward(x) #PREDICTION = fwd pass
    l = loss(y,y_pred)  #LOSS
    l.backward() #GRADIENT  = bwd pass
    with torch.no_grad():
        w -= lr * w.grad #WEIGHT UPDATES
    w.grad.zero_() # make gradient zero
    if e % 10 == 0:
        print('epoch {0}: w = {1}, loss = {2}'.format(e + 1,w,l))
print('prediction after training: f(5) = {0}'.format(forward(5)))
print('prediction after training: f(8) = {0}'.format(forward(8))) 
print('*' * 20)    

#---------------------------------------------------------------------------------------------------------------------------------------


# 3: PREDICTION [manually] , GRADIENT COMPUTATION [autograd] , LOSS COMPUTATION [pytorch loss], PARAMETER UPDATES [pytorch optimizer]

'''General steps to optimiation on pytorch:
1. design model (input,output size, fwd pass)
2. consruct loss and optimizer
3. training loop

- fwd pass : compute prediction
- bwd pass : gradients
- update weights'''

import torch
import torch.nn as nn # import neural n/w module

# f = w *x
# y = 2 *x

x = torch.tensor([1,2,3,4],dtype = torch.float32)
y = torch.tensor([2,4,6,8],dtype = torch.float32)

w  = torch.tensor(0.0,dtype = torch.float32, requires_grad = True)

# model prediction
def forward(x):
    return w*x   # y_pred

print('prediction before training: f(5) = {0}'.format(forward(5)))
print('prediction before training: f(8) = {0}'.format(forward(8)))

# training
lr = 0.01
epoch = 100

loss = nn.MSELoss() # define loss using pytorch
optimizer = torch.optim.SGD([w],lr = lr) # define optimizer using pytorch

for e in range(epoch):
    y_pred = forward(x) #PREDICTION = fwd pass
    l = loss(y,y_pred)  #LOSS
    l.backward() #GRADIENT  = bwd pass
    optimizer.step() # will do optimization step
    optimizer.zero_grad() # make gradient zero
    if e % 10 == 0:
        print('epoch {0}: w = {1}, loss = {2}'.format(e + 1,w,l))
print('prediction after training: f(5) = {0}'.format(forward(5)))
print('prediction after training: f(8) = {0}'.format(forward(8))) 
print('*' * 20)    
#---------------------------------------------------------------------------------------------------------------------------------------

# 4: PREDICTION [pytorch model] , GRADIENT COMPUTATION [autograd] , LOSS COMPUTATION [pytorch loss], PARAMETER UPDATES [pytorch optimizer]

import torch
import torch.nn as nn # import neural n/w module

# f = w *x
# y = 2 *x

x = torch.tensor([[1],[2],[3],[4]],dtype = torch.float32)
y = torch.tensor([[2],[4],[6],[8]],dtype = torch.float32)

x_test = torch.tensor([5],dtype = torch.float32)
n_samples, n_features = x.shape
print("n_samples:{0}, n_features:{1}".format(n_samples, n_features))

input_size = n_features
output_size = n_features
model = nn.Linear(input_size,output_size)

print('prediction before training: f(5) = {0}'.format(model(x_test).item()))

# training
lr = 0.01
epoch = 1000

loss = nn.MSELoss() # define loss using pytorch
optimizer = torch.optim.SGD(model.parameters(),lr = lr) # define optimizer using pytorch

for e in range(epoch):
    y_pred = model(x) #PREDICTION = fwd pass
    l = loss(y,y_pred)  #LOSS
    l.backward() #GRADIENT  = bwd pass
    optimizer.step() # will do optimization step
    optimizer.zero_grad() # make gradient zero
    if e % 100 == 0:
        [w,b]  = model.parameters()
        print('epoch {0}: w = {1}, loss = {2}'.format(e + 1,w[0][0].item(),l))
print('prediction after training: f(5) = {0}'.format(model(x_test).item()))

print('*' * 20)
#---------------------------------------------------------------------------------------------------------------------------------------

# 4[using class]: PREDICTION [pytorch model] , GRADIENT COMPUTATION [autograd] , LOSS COMPUTATION [pytorch loss], PARAMETER UPDATES [pytorch optimizer]

import torch
import torch.nn as nn # import neural n/w module

# f = w *x
# y = 2 *x

x = torch.tensor([[1],[2],[3],[4]],dtype = torch.float32)
y = torch.tensor([[2],[4],[6],[8]],dtype = torch.float32)

x_test = torch.tensor([5],dtype = torch.float32)
n_samples, n_features = x.shape
print("n_samples:{0}, n_features:{1}".format(n_samples, n_features))

input_size = n_features
output_size = n_features

class LinearRegression(nn.Module):

    def __init__(self,input_dim,output_dim):
        super(LinearRegression,self).__init__()
        # define layers
        self.lin = nn.Linear(input_dim,output_dim)

    def forward(self,x):
        return self.lin(x)

model = LinearRegression(input_size,output_size)

print('prediction before training: f(5) = {0}'.format(model(x_test).item()))

# training
lr = 0.01
epoch = 1000

loss = nn.MSELoss() # define loss using pytorch
optimizer = torch.optim.SGD(model.parameters(),lr = lr) # define optimizer using pytorch

for e in range(epoch):
    y_pred = model(x) #PREDICTION = fwd pass
    l = loss(y,y_pred)  #LOSS
    l.backward() #GRADIENT  = bwd pass
    optimizer.step() # will do optimization step
    optimizer.zero_grad() # make gradient zero
    if e % 100 == 0:
        [w,b]  = model.parameters()
        print('epoch {0}: w = {1}, loss = {2}'.format(e + 1,w[0][0].item(),l))
print('prediction after training: f(5) = {0}'.format(model(x_test).item()))

print('*' * 20)    
#---------------------------------------------------------------------------------------------------------------------------------------

