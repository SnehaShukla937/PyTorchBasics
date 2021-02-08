'''General steps to optimiation on pytorch:
1. design model (input,output size, fwd pass)
2. consruct loss and optimizer
3. training loop
- fwd pass : compute prediction
- bwd pass : gradients
- update weights'''

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 1] PREPARE DATA
x_numpy,y_numpy = datasets.make_regression(n_samples = 100,n_features = 1,noise = 20,random_state = 1)

x = torch.from_numpy(x_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0],1)

n_samples,n_features = x.shape

# 2] MODEL
ip_size = n_features
op_size = 1

model = nn.Linear(ip_size,op_size)

# 3] LOSS AND OPTIMIZER
lr = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr = lr)

# 4] TRAINING
epoch = 250
for e in range(epoch):
    # FWD PASS
    y_pred = model(x)
    loss = criterion(y_pred,y)
    # BWD PASS : GRADIENT CALCULATION
    loss.backward()
    # UPDATE WEIGHT
    optimizer.step()
    # MAKE EMPTY GRADIENTS
    optimizer.zero_grad()

    if (e+1) % 25 == 0:
        print("Epoch {0}: loss {1}".format(e+1,loss.item()))

# plotting
pred = model(x).detach().numpy()
plt.plot(x_numpy,y_numpy,'ro')
plt.plot(x_numpy,pred,'b')
plt.show()
