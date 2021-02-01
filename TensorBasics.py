# to check cuda is available with pytorch

import torch
print(torch.cuda.is_available())
print('*'*12)
# Pytorch : a python library and ML , DL framework. used for handlling tensor related opertions.
# Tensor : 1d,2d,3d or multidimensional.

# 1.CREATE EMPTY TENSORS [ ARG.--TENSOR_SIZE ]
print('create empty tensors......')
x = torch.empty(1) # 1d (size 1 tensor)
y = torch.empty(5) # 1d (size 5)
z = torch.empty(2,3) # 2d (2 rows, 3 col)
a = torch.empty(2,2,3) # 3d 
print(x)
print(y)
print(z)
print(a)
print('*'*12)

# 2.CREATE RANDOM NO., ONES, ZEROS TENSORS [ ARG.--TENSOR_SIZE ]
r = torch.rand(2,3)
print('random:\n',r)
r = torch.zeros(2,3)
print('zeros:\n',r)
r = torch.ones(2,3)
print('ones:\n',r)
print('*'*12)

# 3.GET DATATYPE , SET DATATYPE IN TENSORS
print(r.dtype)
r = torch.rand(2,3,dtype = torch.double)
print(r)
print(r.dtype)
r = torch.ones(2,3,dtype = torch.float)
print(r)
print(r.dtype)
print('*'*12)

# 4.GET SIZE IN TENSOR
print(r.size())
print('*'*12)

# 5.CONSTRUCT TENSOR FROM DATA
x = torch.tensor([2.3,8,9,1.9])
print(x,x.dtype,x.size())
print('*'*12)

# 6.BASIC OPERATIONS ON TENSORS
x = torch.rand(2,2)
y = torch.rand(2,2)
print(x,y)
''' _ (underscore) always perform inplace operation in pytorch.'''

# 6a.add
z1 = x + y # using '+'
z2 = torch.add(x,y) # using fn
print("using operator:\n",z1,"\nusing fn:\n",z2)
y.add_(x)   # using inplace addition
print("inplace addition:\n",y)
print('*'*12)

# 6b.sub
x = torch.rand(2,2)
y = torch.rand(2,2)
print(x,y)
z1 = x - y # using '-'
z2 = torch.sub(x,y) # using fn
print("using operator:\n",z1,"\nusing fn:\n",z2)
y.sub_(x)   # using inplace subtraction
print("inplace subtraction:\n",y)
print('*'*12)

# 6c.mul
x = torch.rand(2,2)
y = torch.rand(2,2)
print(x,y)
z1 = x * y # using '*'
z2 = torch.mul(x,y) # using fn
print("using operator:\n",z1,"\nusing fn:\n",z2)
y.mul_(x)   # using inplace mul
print("inplace mul:\n",y)
print('*'*12)

# 6d.div
x = torch.rand(2,2)
y = torch.rand(2,2)
print(x,y)
z1 = x / y # using '*'
z2 = torch.div(x,y) # using fn
print("using operator:\n",z1,"\nusing fn:\n",z2)
y.div_(x)   # using inplace div (y/x)
print("inplace div:\n",y)
print('*'*12)

# 7.SLICING OPERATION IN TENSORS
x = torch.rand(5,3)
print("for slicing:\n",x)
print(x[1,1],x[2,:],x[:,2],x[:,0])
print('*'*12)

# using .item() in slicing to get actual value: .item() only gives one value (or) gives scaler value.
print('.item() values:')
print(x[1,1].item())
print(x[3,2].item())
print(x[0,1].item())
print('*'*12)

# 8.RESHAPING A TENSOR (shape should be compatible)
print('reshaping:')
y = x.view(15) 
print(y,y.size())
y = x.view(3,-1) 
print(y,y.size())
y = x.view(5,-1,3) 
print(y,y.size())
y = x.view(-1,15) 
print(y,y.size())
print('*'*12)

# 9.CONVERT NUMPY TO TENSOR AND TENSOR TO NUMPY
import numpy as np
''' numpy only handles "cpu" operation not "gpu".
Any modification on tensor(or numpy array) will affect the coverted numpy array(or tensor) when torch device is set as "cpu". '''
# from torch to numpy
a = torch.ones(5)
print(a,type(a))
b = a.numpy()
print(b,type(b))

a.add_(1) # modify a(on torch)
print(a)
print(b)

# from numpy to torch
x = np.ones(5)
print(x,type(x))
y = torch.from_numpy(x)
print(y,type(y))

x += 1 # modify x(on numpy)
print(x)
print(y)

# if torch is running on gpu:
if torch.cuda.is_available():
    device = torch.device("cuda") # set device
    x = torch.ones(5,device = device)
    y = torch.ones(5)
    y = y.to(device) # set device for y tensor (as it is not set in previous line)
    z = x + y
    z = z.to("cpu") # z is considered as numpy so device set to "cpu"
    print(z)
print('*'*12)

# 10. SET REQUIRES_GRAD IN TENSORS
x = torch.tensor([2],requires_grad = True,dtype = torch.float32) # we need derivative of x
print(x)
y = 4*x + 2*x**2
y.backward()
print(x.grad) # will give dy/dx value (4 + 8 = 12)
print('*'*12)
