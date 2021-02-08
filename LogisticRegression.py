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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# 1] PREPARE DATA

# load data and get data and target values
data = datasets.load_breast_cancer()
x,y = data.data,data.target
# get shape of input
n_samples,n_features = x.shape 
# split data (80%-20%)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)
# feature scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
# convert into tensors
x_train = torch.from_numpy(x_train.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))
# reshape taget tensor
y_train = y_train.view(y_train.shape[0],1)
y_test = y_test.view(y_test.shape[0],1)

# 2] MODEL

'''f = wx + b, sigmoid at the end'''
class LogisticRegression(nn.Module):

    def __init__(self,n_input_features):
        super(LogisticRegression,self).__init__()
        self.linear = nn.Linear(n_input_features,1)

    def forward(self,x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = LogisticRegression(n_features)

# 3] LOSS AND OPTIMIZER
lr = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(),lr = lr)

# 4] TRAINING
epoch = 100
for e in range(epoch):
    # fwd pass and loss cal
    y_pred = model(x_train)
    loss = criterion(y_pred,y_train)
    # bwd pass
    loss.backward()
    # updates
    optimizer.step()
    # make empty gradient
    optimizer.zero_grad()

    if (e + 1)% 10 == 0:
        print('epoch {0}: loss = {1}'.format(e+1,loss.item()))

with torch.no_grad():
    y_predicted = model(x_test)
    #print(y_predicted)
    y_predicted_class = y_predicted.round()
    #print(y_predicted_class)
    acc = y_predicted_class.eq(y_test).sum() / float(y_test.shape[0])
    print("accuracy :{0}".format(acc))
