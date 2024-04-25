import numpy as np
import matplotlib.pyplot as plt
import os
import pdb

import torch

# make additiones like integers
a = torch.tensor(3)
b = torch.tensor(2)
c = a + b

# turn to numpy array
a = np.array([1,2,3,4])
b = torch.tensor(a) #Change np.array into torch.tensor
c = b.numpy() #Change torch.tensor into np.array

try:
  m = torch.multiply(a,b)
  print('It seems that you can multiply numpy arrays with torch tensors.')
except:
  print("See? You can't multiply a numpy array with a tensor")

# memorize in CPU or GPU
a = np.array([1,0,2])
b = torch.tensor(a)
print(b.device) # default storage device is CPU
print(b)


# We can change the device of a tensor by using the to() method:
if torch.cuda.is_available(): #We first check if a GPU is available:
    c = b.to("cuda")   # ! Need to install CUDA for this to work
    print('We changed the device')


print(c.device)
print(c)



# we GPU usage (cuda) is 1000 times faster than CPU
import time
n = 4096
f = 1
b = torch.randn(f*n,f*n)
c = b.cuda()

def power(x):
  return x**2

start_time = time.time()
b_ = power(b)
end_time = time.time()
print('on cpu, time elapsed:', end_time-start_time, 'seconds')

start_time = time.time()
c = power(c)
end_time = time.time()
print('on cuda, time elapsed:', end_time-start_time, 'seconds')




# How to use PyTorch to do automatic differentiation, gradient calculation













































