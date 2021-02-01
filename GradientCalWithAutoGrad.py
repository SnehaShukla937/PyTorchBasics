# Gradient is very important for model optimization. Pytorch provides autograd package which can do all the gradient calculation.

import torch

# 1. CALCULATE GRADIENTS IN PYTORCH
x = torch.randn(3,requires_grad = True) # set requires_grad = True
print(x)

''' forward pass : calculate y
    backward pass : calculate gradient dy/dx  [y has a grad_fn = <AddBackward>]'''

y = x + 2 # pytorch creates computational graph (input = x,2 , operator = + , output = y)
print(y) # grad_fn = <AddBackward>

z = y*y*2
print(z) #grad_fn = <MulBackward>

z = z.mean()
print(z) # grad_fn = <MeanBackward>

z.backward() # dz/dx # will calculate grad_fn for x (or gradient of z wrt x) (as x requires_grad = true)
print(x.grad)     # x has attribute '.grad' , where gradient are stored.
print("*"*12)

# 2. STOP PYTORCH FROM CREATING GRADIENT FUNCTION OR FROM TRACKING HISTORY IN COMPUTATIONAL GRAPH  [3 ways]

x = torch.randn(4,requires_grad = True) # set requires_grad = True
print(x)
# method 1. inplace modification (_)
x.requires_grad_(False)
print(x)

# method 2. create new tensor
y = x.detach()
print(y)

# method 3. 'with' operation
z = torch.rand(2,requires_grad = True)
print(z)
with torch.no_grad():
    y = z + 2
    print(y)
print("*"*12)

# 3. DUMMY TRAINING EXAMPLE : Always set weight grad to zero before next epoch
'''Everytime when gradients are calculated it will stored in grad and accumulated (or added) in next epoch. which is wrong for optimization.
   we need to update the weight (by increment or decrement based on loss value) ...but here previous weight is added to new weight.
   To avoid this we just empty the gradient (making weights = 0) before proceed to next epoch. '''

weights  = torch.ones(3,requires_grad = True)
print(weights)
for epoch in range(3):
    model_op = (weights*3).sum()
    model_op.backward()
    print(weights.grad)
    weights.grad.zero_() # set weights grad to zero.
print("*"*12)
